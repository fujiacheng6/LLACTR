# bash encode.sh

for lora_weights in "model/gift/sl_1000/checkpoint-32" "model/gift/sl_1000/checkpoint-65" "model/gift/sl_1000/checkpoint-98"
do
    python encode.py \
        --prompt_path "llm_emb/gift/field.txt" \
        --llm_emb_path "${lora_weights}/field_sl_eos.pt" \
        --lora_weights $lora_weights \
        --base_model "./Llama3_Checkpoints" 
done