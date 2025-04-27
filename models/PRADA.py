#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 
from .prompt import Prompt 



class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
       

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        if configs.pretrained == True:
           
          
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            original_params = sum(p.numel() for p in self.gpt2.parameters())
            config = LoraConfig(
                # task_type=TaskType.CAUSAL_LM, # causal language model
                r=configs.lora_params,
                lora_alpha=4,
                # target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="lora_only",               # bias, set to only lora layers to train
                # modules_to_save=["classifier"],
            )
            self.gpt2 = get_peft_model(self.gpt2, config)
            lora_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
            reduced_params = original_params - lora_params
            print(f"Original parameters: {original_params}")
            print(f"LoRA parameters: {lora_params}")
            print(f"Reduced parameters: {reduced_params}")
            
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name  or 'wpe' in name:   #or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False

    
        self.in_layer_trend = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer_seasonal = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer_residual = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)) , configs.pred_len)
        
        self.prompt_pool_trend = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight,
                mode='trend', token_embedding=self.gpt2.wte, gpt_model=self.gpt2, dataset_name=configs.data, freq=configs.freq, sea_pattern=configs.seasonal_patterns)
        
        self.prompt_pool_seasonal = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight,
                mode='seasonal', token_embedding=self.gpt2.wte, gpt_model=self.gpt2, dataset_name=configs.data, freq=configs.freq, sea_pattern=configs.seasonal_patterns)
                
        self.prompt_pool_residual = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight,
                mode='residual', token_embedding=self.gpt2.wte, gpt_model=self.gpt2, dataset_name=configs.data, freq=configs.freq, sea_pattern=configs.seasonal_patterns)

        
        for layer in (self.gpt2, self.in_layer_trend, self.in_layer_seasonal, self.in_layer_residual, self.prompt_pool_trend, self.prompt_pool_seasonal, self.prompt_pool_residual, self.out_layer):       
            layer.cuda()
            layer.train()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out, res = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], res  # [B, L, D]
    

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
            
        B, L, M = x_enc.shape   # B: batch size, L: sequence length, M: feature length
            
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
        torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
 
        x = rearrange(x_enc, 'b l m -> (b m) l') 

        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c = 3)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_trend, x_seasonal, x_residual = torch.split(x, 1, dim=1)
        x_trend = x_trend.squeeze(1)
        x_seasonal = x_seasonal.squeeze(1)
        x_residual = x_residual.squeeze(1)
        pre_prompted_embedding_trend = self.in_layer_trend(x_trend.float())
        pre_prompted_embedding_seasonal = self.in_layer_seasonal(x_seasonal.float())
        pre_prompted_embedding_residual = self.in_layer_residual(x_residual.float())

        outs_trend = self.prompt_pool_trend(pre_prompted_embedding_trend)
        prompted_embedding_trend = outs_trend['prompted_embedding']
        sim_trend = outs_trend['similarity']
        prompt_key_trend = outs_trend['prompt_key']
        batched_prompt_trend = outs_trend['batched_prompt']
        x_embed_trend = outs_trend['x_embed']
        simlarity_loss_trend = outs_trend['reduce_sim']

        outs_seasonal = self.prompt_pool_seasonal(pre_prompted_embedding_seasonal)
        prompted_embedding_seasonal = outs_seasonal['prompted_embedding']
        sim_seasonal = outs_seasonal['similarity']
        prompt_key_seasonal = outs_seasonal['prompt_key']
        batched_prompt_seasonal = outs_seasonal['batched_prompt']
        x_embed_seasonal = outs_seasonal['x_embed']
        simlarity_loss_seasonal = outs_seasonal['reduce_sim']
    
        outs_residual = self.prompt_pool_residual(pre_prompted_embedding_residual)
        prompted_embedding_residual = outs_residual['prompted_embedding']
        sim_residual = outs_residual['similarity']
        prompt_key_residual = outs_residual['prompt_key']
        batched_prompt_residual = outs_residual['batched_prompt']
        x_embed_residual = outs_residual['x_embed']
        simlarity_loss_residual = outs_residual['reduce_sim']
        
        prompted_embedding = prompted_embedding_trend + prompted_embedding_seasonal + prompted_embedding_residual
        simlarity_loss = simlarity_loss_trend + simlarity_loss_seasonal + simlarity_loss_residual

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*M*3, -1))
            
            
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss

        
        outputs = outputs * stdev[:,:,:M]
        outputs = outputs + means[:,:,:M]
        
        
        def orthogonal_constraint(x1, x2, x3):
            
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x3 = x3.view(x3.size(0), -1)

            inner_product_12 = torch.sum(x1 * x2, dim=1)
            inner_product_13 = torch.sum(x1 * x3, dim=1)
            inner_product_23 = torch.sum(x2 * x3, dim=1)

            loss = torch.mean(inner_product_12**2 + inner_product_13**2 + inner_product_23**2)
            
            return loss
        
        orthogonal_loss = orthogonal_constraint(batched_prompt_trend, batched_prompt_seasonal, batched_prompt_residual)
        res['orthogonal_loss'] = orthogonal_loss 

        return outputs, res




    










