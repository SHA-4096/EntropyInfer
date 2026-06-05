from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig

import torch


def load_model_and_tokenizer(path, model_name, device, compress=False, attn_impl="flash_attention_2", cache_dir=None):
    if "pangu" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
            attn_implementation=attn_impl, 
        )
    else:
        raise ValueError(f"Model {model_name} not supported yet!")
    model = model.eval()
    return model, tokenizer