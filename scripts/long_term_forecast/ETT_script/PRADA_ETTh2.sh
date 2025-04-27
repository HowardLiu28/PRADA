export CUDA_VISIBLE_DEVICES=1

model_name=PRADA

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/ETT-small \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_96 \
  --model $model_name \
  --data ETTh2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.0001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --batch_size 32 \
  --sim_coef -0.05 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --trend_length 96 \
  --seasonal_length 96 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/ETT-small \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_192 \
  --model $model_name \
  --data ETTh2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.0001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --batch_size 32 \
  --sim_coef -0.05 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --trend_length 96 \
  --seasonal_length 12 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/ETT-small \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_336 \
  --model $model_name \
  --data ETTh2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.0001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --batch_size 32 \
  --sim_coef -0.05 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --trend_length 96 \
  --seasonal_length 12 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/ETT-small \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_720 \
  --model $model_name \
  --data ETTh2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 4 \
  --batch_size 32 \
  --sim_coef -0.05 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --trend_length 24 \
  --seasonal_length 24 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \