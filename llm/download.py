from huggingface_hub import snapshot_download
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 1. 修改模型ID ---
repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# --- 2. 修改保存路径 (强烈建议！避免新旧模型混在一起) ---
local_dir = "./Llama3_Checkpoints"

print(f"开始下载小模型 {repo_id} 到 {local_dir}")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print("小模型下载完成！")