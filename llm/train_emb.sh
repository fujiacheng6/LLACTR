# nohup bash train_emb.sh > ./logs/train_emb_gift_sl_1000.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=29501 train_emb.py \
    --base_model "./Llama3_Checkpoints" \
    --train_data_path '["../data/ml1m/instruction/train.json"]'  \
    --val_data_path '["../data/ml1m/instruction/valid.json"]'  \
    --output_dir './model/ml1m/sl_1000' \
    --batch_size 32 \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --cutoff_len 2048 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --group_by_length \
    --seed 2024

