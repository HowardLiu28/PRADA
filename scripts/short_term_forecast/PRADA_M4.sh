export CUDA_VISIBLE_DEVICES=1

model_name=PRADA

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --d_model 768 \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 64 \
  --eval_batch_size 8 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --train_epochs 50 \
  --model_comment 'PRADA-M4' \
  --patience 3 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --sim_coef -0.1 \
  --orthogonal_coef 0.0 \
  --pool_size  128 \
  --trend_length 4 \
  --seasonal_length 2 \
  --lora_params 512 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --d_model 768 \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 64 \
  --eval_batch_size 8 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --train_epochs 50 \
  --model_comment 'PRADA-M4' \
  --patience 3 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --sim_coef -0.1 \
  --orthogonal_coef 0.0 \
  --pool_size  128 \
  --trend_length 1 \
  --seasonal_length 1 \
  --lora_params 512 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model $model_name \
  --d_model 768 \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 64 \
  --eval_batch_size 8 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --train_epochs 50 \
  --model_comment 'PRADA-M4' \
  --patience 3 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --sim_coef -0.1 \
  --orthogonal_coef 0.0 \
  --pool_size  128 \
  --trend_length 4 \
  --seasonal_length 2 \
  --lora_params 512 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --d_model 768 \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 64 \
  --eval_batch_size 8 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --train_epochs 50 \
  --model_comment 'PRADA-M4' \
  --patience 3 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --sim_coef -0.1 \
  --orthogonal_coef 0.0 \
  --pool_size  128 \
  --trend_length 4 \
  --seasonal_length 2 \
  --lora_params 512 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model $model_name \
  --d_model 768 \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 64 \
  --eval_batch_size 8 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --train_epochs 50 \
  --model_comment 'PRADA-M4' \
  --patience 3 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --sim_coef -0.1 \
  --orthogonal_coef 0.0 \
  --pool_size  128 \
  --trend_length 4 \
  --seasonal_length 2 \
  --lora_params 512 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model $model_name \
  --d_model 768 \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 64 \
  --eval_batch_size 8 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --train_epochs 50 \
  --model_comment 'PRADA-M4' \
  --patience 3 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --sim_coef -0.1 \
  --orthogonal_coef 0.0 \
  --pool_size  128 \
  --trend_length 4 \
  --seasonal_length 2 \
  --lora_params 512 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \