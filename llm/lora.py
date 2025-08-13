from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import PeftModel
import torch, gc, time, textwrap, re, os


BASE_MODEL   = "./Llama3_Checkpoints"
ADAPTER_PATH = "./model/gift/sl_1000"       

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"
VOCAB_SIZE = len(tokenizer)# 32001 after <pad>


def greedy(model, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

def build_prompt(feature: str) -> str:
    return (f"### Instruction:\n"
            f"Given a feature, please answer which field it belongs to. "
            f"Please select the field name from the candidate set.\n\n"
            f"### Input:\n"
            f"The feature is: {feature}.\n"
            f"The candidate set is: ['UserID', 'Gender', 'Age', "
            f"'Occupation', 'MovieID', 'Title', 'Genres', 'Timestamp'].\n\n"
            f"### Response:\n")

FIELD_NAMES = ['UserID', 'Gender', 'Age', 'Occupation',
               'MovieID', 'Title', 'Genres', 'Timestamp']
TEST_CASES = [
    ("795",  "UserID"),
    ("M",    "Gender"),
    ("Action|Adventure|Thriller", "Genres"),
    ("Star Wars: Episode IV - A New Hope (1977)", "Title"),
    ("25",   "Age"),
]

def evaluate_full_outputs(model, tag: str, max_new_tokens: int = 50) -> None:
    """Print full prompts and full model outputs for manual inspection.

    This bypasses automatic accuracy to help diagnose evaluation issues.
    """
    for idx, (feat, exp) in enumerate(TEST_CASES, 1):
        prompt = build_prompt(feat)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_output = tokenizer.decode(out[0], skip_special_tokens=True)

        print("-" * 20, f"Case {idx}", "-" * 20)
        print(f"Feature: '{feat}'")
        print(f"Expected: '{exp}'")
        print("\nPrompt:\n" + prompt)
        print("\n==> FULL MODEL OUTPUT:\n" + full_output + "\n")

        lower_out = full_output.lower()
        detected = [f for f in FIELD_NAMES if f.lower() in lower_out]
        print(f"Detected fields in output: {detected}\n")

def evaluate(model, tag, verbose=False):
    correct = 0
    for idx, (feat, exp) in enumerate(TEST_CASES, 1):
        ans_raw = greedy(model, build_prompt(feat))
        # pick first field name appearing in answer
        ans = next((f for f in FIELD_NAMES if f.lower() in ans_raw.lower()),
                   ans_raw.split()[0] if ans_raw.split() else ans_raw)

        hit = (exp.lower() == ans.lower())
        correct += int(hit)

        if verbose:
            print(f"[{tag}] Case {idx}: feature='{feat}'  expected='{exp}'")
            print("  raw  :", textwrap.shorten(ans_raw.replace('\\n', '\\\\n').replace('\n', '\\n'), width=120))
            print(f"  extracted ⇒ '{ans}'   {'y' if hit else 'n'}\n")

    print(f"{tag} accuracy: {correct}/{len(TEST_CASES)} = {correct/len(TEST_CASES)*100:.1f}%")
    return correct


print("Loading base model (8‑bit)…")
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True,
                             llm_int8_enable_fp32_cpu_offload=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_cfg,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
base.resize_token_embeddings(VOCAB_SIZE)
base.config.pad_token_id = tokenizer.pad_token_id
base.eval()

# Print full outputs for base model instead of computing accuracy
evaluate_full_outputs(base, "Base")

# clean
del base; gc.collect(); torch.cuda.empty_cache()

# ─── Load LoRA model (if provided) ───────────────────────────────────────────
if ADAPTER_PATH and os.path.isdir(ADAPTER_PATH):
    print("Loading LoRA‑augmented model…")
    parent = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="cpu",
        torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        trust_remote_code=True)
    parent.resize_token_embeddings(VOCAB_SIZE)
    parent.config.pad_token_id = tokenizer.pad_token_id

    # ---- Strip embed/lm_head weights from adapter state_dict to avoid size mismatch ----
    from tempfile import TemporaryDirectory
    import shutil, torch
    tmp_adapter = ADAPTER_PATH
    for fname in ("adapter_model.bin", "adapter_model.safetensors"):
        path = os.path.join(ADAPTER_PATH, fname)
        if os.path.isfile(path):
            tmp_dir = TemporaryDirectory()
            tmp_adapter = tmp_dir.name
            new_path = os.path.join(tmp_adapter, fname)
            if fname.endswith(".bin"):
                sd = torch.load(path, map_location="cpu")
            else:
                from safetensors.torch import load_file, save_file
                sd = load_file(path, device="cpu")
            for k in ["base_model.model.model.embed_tokens.weight",
                      "base_model.model.lm_head.weight"]:
                sd.pop(k, None)
            if fname.endswith(".bin"):
                torch.save(sd, new_path)
            else:
                save_file(sd, new_path)
            # copy adapter_config.json
            shutil.copy2(os.path.join(ADAPTER_PATH, "adapter_config.json"), tmp_adapter)
            break
    lora = PeftModel.from_pretrained(parent, tmp_adapter, device_map="auto")
    lora.to("cuda"); lora.eval()
    # Print full outputs for LoRA model instead of computing accuracy
    evaluate_full_outputs(lora, "LoRA")

    del lora; gc.collect(); torch.cuda.empty_cache()
else:
    print("No LoRA adapter found – skipped.")