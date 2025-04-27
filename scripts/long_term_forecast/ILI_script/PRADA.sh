export CUDA_VISIBLE_DEVICES=1

model_name=PRADA

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/illness \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model $model_name \
  --data ili \
  --number_variable 7 \
  --features M \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 24 \
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
  --trend_length 36 \
  --seasonal_length 24 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/illness \
  --data_path national_illness.csv \
  --model_id ili_36_36 \
  --model $model_name \
  --data ili \
  --number_variable 7 \
  --features M \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 36 \
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
  --trend_length 36 \
  --seasonal_length 24 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/illness \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data ili \
  --number_variable 7 \
  --features M \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 48 \
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
  --trend_length 36 \
  --seasonal_length 24 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./all_datasets/illness \
  --data_path national_illness.csv \
  --model_id ili_36_60 \
  --model $model_name \
  --data ili \
  --number_variable 7 \
  --features M \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 60 \
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
  --trend_length 36 \
  --seasonal_length 24 \
  --lora_params 1024 \
  --auxi_loss 'MAE' \
  --auxi_coef 1 \
  --module_first True \