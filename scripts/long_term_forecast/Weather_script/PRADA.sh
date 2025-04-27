export CUDA_VISIBLE_DEVICES=1

model_name=PRADA

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/weather \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 2 \
  --batch_size 256 \
  --sim_coef -0.1 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --period 24 \
  --trend_length 96 \
  --seasonal_length 48 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/weather \
  --data_path weather.csv \
  --model_id weather_512_192 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 192 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 2 \
  --batch_size 256 \
  --sim_coef -0.1 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --period 24 \
  --trend_length 96 \
  --seasonal_length 48 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/weather \
  --data_path weather.csv \
  --model_id weather_512_336 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 336 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 2 \
  --batch_size 256 \
  --sim_coef -0.1 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --period 24 \
  --trend_length 96 \
  --seasonal_length 48 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/weather \
  --data_path weather.csv \
  --model_id weather_512_720 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 720 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --add_trainable_prompt 4 \
  --prompt_length 2 \
  --batch_size 256 \
  --sim_coef -0.1 \
  --orthogonal_coef 5.0 \
  --pool_size 128 \
  --percent 100 \
  --period 24 \
  --trend_length 96 \
  --seasonal_length 48 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \