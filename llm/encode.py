from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os
import math
import json
import argparse
import pandas as pd
import json
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from accelerate import Accelerator
from accelerate.utils import gather_object

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
accelerator = Accelerator()
print(f"cuda num: {torch.cuda.device_count()}")
transformers.set_seed(42)
parser = argparse.ArgumentParser()

batch_size = 1 # 32

parser.add_argument('--prompt_path', type=str, default=f"")
parser.add_argument('--llm_emb_path', type=str, default=f"")
parser.add_argument('--llm_emb_path2', type=str, default=f"")   
parser.add_argument('--lora_weights', type=str, default="")     
parser.add_argument('--base_model', type=str, default="") 

args = parser.parse_args()
print('-'*120)
for key, value in vars(args).items():
    print(f"{key}: {value}")
print('-'*120)

prompt_path = args.prompt_path
llm_emb_path = args.llm_emb_path
llm_emb_path2 = args.llm_emb_path2
lora_weights = args.lora_weights
base_model = args.base_model

# loading prompt
with open(prompt_path, 'r') as file: # encoding='ISO-8859-1'
    content_list = file.readlines()
    all_promt_list = [line.strip() for line in content_list]

if accelerator.is_main_process:
    print(f"all_promt_list = {(all_promt_list[:3])}")
    print(f"all_promt_list_len = {len(all_promt_list)}")

if torch.cuda.device_count() > 0:
    device_map = {"": accelerator.process_index}
else:
    device_map = "cpu"

# loading llama
if lora_weights == "":
    base_model = base_model # -Instruct
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print('len(tokenizer)=', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"    
else:
    print('load checkpoint...')
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache = False
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    print('len(tokenizer)=', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"    

    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache = False
    )
    model.merge_and_unload()



# encoding prompt
model.eval()

def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

with accelerator.split_between_processes(all_promt_list) as promt_list:
    print(f"accelerator.process_index: {accelerator.device}; promt_list: {len(promt_list)}")
    predict_embeddings = []
    predict_embeddings2 = []
    for i, batch_input in tqdm(enumerate(batch(promt_list, batch_size)), total=len(promt_list)//batch_size):
        input = tokenizer(batch_input, return_tensors="pt", padding=True).to(model.device) # , truncation=True , max_length=max_length
        input_ids = input.input_ids
        attention_mask = input.attention_mask

        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False)
        hidden_states = outputs.hidden_states # hidden_states: 1+32  torch.Size([batch_size, sequence_length, 4096])

        emb_eos = hidden_states[-1][:, -1, :].detach().cpu()
        predict_embeddings.append(emb_eos)

        emb_meanpooling = torch.mean(hidden_states[-1].detach().cpu(), dim=1)       
        predict_embeddings2.append(emb_meanpooling)

all_predict_embeddings = gather_object(predict_embeddings) # list

# save embeddings
if accelerator.is_main_process:
    all_predict_embeddings = torch.cat(all_predict_embeddings, dim=0) # tensor

    print(f"all_predict_embeddings = {all_predict_embeddings.shape}")

    torch.save(all_predict_embeddings.float(), llm_emb_path)

