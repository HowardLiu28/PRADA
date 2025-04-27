import torch
import torch.nn as nn
import os
from transformers import GPT2Tokenizer, GPT2Model

class PromptLearner(nn.Module):
    prompt_root = "prompt_bank"
    freq_info_bank = {
        'h': "hourly",
        'd': "daily",
        'w': "weekly",
        'm': "monthly",
        's': "secondly",
        't': "minutely"
    }
    inchannel_bank = {
        'Yearly': 13,
        'Quarterly': 17,
        'Monthly': 37,
        'Weekly': 27,
        'Daily': 29,
        'Hourly': 97
    }
    def __init__(self, token_embedding, mode, dataset_name, freq, sea_pattern, dtype=torch.float32):
        super(PromptLearner, self).__init__()
        self.mode = mode
        self.dtype = dtype
        self.token_embedding_weight = token_embedding.weight.cuda()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', trust_remote_code=True, local_files_only=True)
        
        self.freq_info = self.freq_info_bank[freq]
        self.inchannel = self.inchannel_bank[sea_pattern]
        
        # Define context initialization based on the mode
        if mode == 'trend':
            self.ctx_init = (
                "The time series shows a X X X X trend, increasing/decreasing over time."
            )
            
        elif mode == 'seasonal':
            self.ctx_init = (
                # f"The time series exhibits a X X X X seasonal pattern, repeating its behavior {self.freq_info}."
                "The time series exhibits a X X X X seasonal pattern, repeating its behavior over time."
            )
        elif mode == 'residual':
            if 'ETT' in dataset_name:
                prompt_file = os.path.join(self.prompt_root, 'ETT.txt')
            elif 'traffic' in dataset_name:
                prompt_file = os.path.join(self.prompt_root, 'Traffic.txt')
            elif 'ECL' in dataset_name:
                prompt_file = os.path.join(self.prompt_root, 'ECL.txt')
            elif 'weather' in dataset_name:
                prompt_file = os.path.join(self.prompt_root, 'Weather.txt')
            elif 'ili' in dataset_name:
                prompt_file = os.path.join(self.prompt_root, 'ILI.txt')
            elif 'm4' in dataset_name:
                prompt_file = os.path.join(self.prompt_root, 'm4.txt')
            else:
                raise NotImplementedError("Dataset not supported")
            with open(prompt_file, 'r', encoding='utf-8') as file:
                line = file.readlines()[0].strip()
                self.ctx_init = line
                
        # If special tokens are defined, add them to the tokenizer and extend embedding matrix
        self.token_embedding = token_embedding.cuda()
        
        # Tokenize the context
        tokenized_prompts = self.tokenizer(self.ctx_init, return_tensors='pt')['input_ids'].cuda()
        with torch.no_grad():
            embeddings = self.token_embedding(tokenized_prompts)
        self.tokenized_prompts = tokenized_prompts
        
        self.embeddings = embeddings
        
        if mode in ['trend', 'seasonal']:
            n_pre_ctx = 5   # The time series shows/exhibits a
            n_ctx = 4   # X X X X
            if 'ili' in dataset_name:
                self.conv1d = nn.Conv1d(in_channels=4, out_channels=n_ctx, kernel_size=1)  # learn X X X X tokens, 4 for input len=36
            elif 'm4' in dataset_name:
                self.conv1d = nn.Conv1d(in_channels=self.inchannel, out_channels=n_ctx, kernel_size=1)  # learn X X X X tokens
            else:
                self.conv1d = nn.Conv1d(in_channels=64, out_channels=n_ctx, kernel_size=1)  # learn X X X X tokens, 64 for input len=512, 12 for input len=96
            
            self.register_buffer("token_prefix", embeddings[:, :n_pre_ctx + 1, :])
            self.register_buffer("token_suffix", embeddings[:, n_pre_ctx + 1 + n_ctx:, :])

    def forward(self, x=None):
        if x is not None:
            learnable_ctx = self.conv1d(x)
            learnable_ctx = torch.mean(learnable_ctx, dim=0).unsqueeze(0)
            return torch.cat(
                [
                    self.token_prefix,
                    learnable_ctx,
                    self.token_suffix
                ], dim=1
            )
        else:
            return self.embeddings

class TextEncoder(nn.Module):
    def __init__(self, model, pool_size=1000):
        super(TextEncoder, self).__init__()
        self.model = model
        self.hidden_size = self.model.config.hidden_size
        self.proj = nn.Linear(1, pool_size)
    
    def forward(self, prompt_embeddings):
        bs, seq_len, emb_dim = prompt_embeddings.shape
        with torch.no_grad():
            outputs = self.model(inputs_embeds = prompt_embeddings)
        last_hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_states, dim=1).permute(1, 0)
        encoded_output = self.proj(pooled_output)
        
        return encoded_output



class Prompt(nn.Module):
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform',wte = None, mode='trend',
                 token_embedding=None, gpt_model=None, dataset_name=None, freq='h', sea_pattern=None):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        print(self.pool_size)
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.wte = wte  # GPT2 wte.weight
        self.mode = mode

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        
        self.prompt_learner = PromptLearner(token_embedding, mode, dataset_name, freq, sea_pattern)
        self.text_encoder = TextEncoder(gpt_model, pool_size=self.pool_size)
        
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        # x_embed: 224*64*768
        out = dict()
        if self.prompt_key:   #if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            if self.mode in ['trend', 'seasonal']:
                prompt_embeddings = self.prompt_learner(x_embed)
            else:
                prompt_embeddings = self.prompt_learner()
            # prompt_embeddings = self.prompt_learner()
            prompt_key = self.text_encoder(prompt_embeddings).t()
            
            prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C   self.prompt_key
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            # batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
                
            batched_prompt_raw = prompt_key[idx] # B, top_k, length, C
            batched_prompt_raw = batched_prompt_raw.unsqueeze(2) # B, top_k, 1, length, C

            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompt_key'] = prompt_key  # prompt_key
        out['batched_prompt'] = batched_prompt
        out['x_embed'] = x_embed

        return out