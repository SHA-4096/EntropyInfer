import os, sys
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import gc

cwd = os.getcwd()
print(f"cwd: {cwd}")
sys.path.append(cwd)


from experiment.tools.compression_config import config_compress
from experiment.tools.model_patching import patch_models
from experiment.tools.load_model_and_tokenizer import load_model_and_tokenizer

import transformers

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", None)




def parse_args():
    parser = argparse.ArgumentParser(description="Longbench Eval")
    parser.add_argument(
        "--mode",
        type=str,
        default="base",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name or path",
        default="pangu-7b-v11",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Config path for Compression Methods",
        default=None
    )
    args = parser.parse_args()
    return args

args = parse_args()

with open("config/model2path.json") as f:
    model2path = json.load(f)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "pangu" in model_name.lower():
        if "pangu-7b" in model_name.lower() and "think" not in model_name.lower():
            prompt = prompt + "/no_think"
            print(f"Added no_think for pangu-7b")
        print('pangu')
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    else:
        raise NotImplementedError(f"Chat building not implemented for {model_name}")
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "pangu-7b" in model_name.lower():
        thinking_content = response.split("[unused17]")[0].split("[unused16]")[-1].strip()
        pred = response.split("[unused17]")[-1].split("[unused10]")[0].strip()
        response = {
            "thinking_content": thinking_content,
            "pred": pred
        }
        return response
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name_or_path, out_path):
    preds = []
    with open(f"{out_path}_tmp", "w", encoding="utf-8") as f:
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name_or_path:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, model_name_or_path)

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]

            custom_past_key_values = transformers.cache_utils.DynamicCache()
            if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    past_key_values = custom_past_key_values,
                    use_cache=True
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=[tokenizer.eos_token_id],
                    past_key_values = custom_past_key_values,
                    use_cache=True
                )[0]

            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            raw_pred = post_process(pred, model_name_or_path)
            thinking_content = None
            if isinstance(pred, dict):
                thinking_content = raw_pred["thinking_content"]
                pred = raw_pred["pred"]
            else:
                pred = raw_pred

            preds.append(pred)

            data_to_dump = {"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}
            if thinking_content is not None:
                data_to_dump["thinking_content"] = thinking_content

            json.dump(data_to_dump, f, ensure_ascii=False )
            f.write('\n')
            f.flush()


            gc.collect()
            torch.cuda.empty_cache()


    with open(out_path, "w", encoding="utf-8") as f:
        for json_obj, pred in zip(data, preds):
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False )
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    rand_seed = 42
    seed_everything(rand_seed)
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_name
    model_id = model2path[model_name]
    # define your model
    max_length = 128000

    datasets = [
                "qasper", "narrativeqa", "multifieldqa_en", # single doc
                "hotpotqa", "2wikimqa", "musique",          # multi doc
                "trec", "triviaqa", "samsum",               # few-shot
                "gov_report", "qmsum", "multi_news",        # sum
                "passage_count", "passage_retrieval_en",    # Synthetic
                "lcc", "repobench-p",                       # code
                ]

    print(datasets)
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # Result folder
    result_folder = "pred_longbench"
    os.makedirs(result_folder, exist_ok=True)

    # build output path
    write_model_name = model_name
    compression_method = args.mode
    compression_config_path = args.config_path
    prefix = f"pred_longbench/{compression_method}"
    compression_file_name = ""
    if compression_config_path is not None:
        compression_file_name = "--".join(compression_config_path.split("/")[2:]).split(".")[0]
    out_path_prefix = f"{prefix}/{write_model_name}_{compression_file_name}"

    # dump config to result_dir
    os.makedirs(out_path_prefix, exist_ok=True)
    compression_config = None
    if args.config_path is not None:
        compression_config = json.load(open(args.config_path, "r"))
    config_save_path = os.path.join(out_path_prefix, "config.json")
    config = {
        "compression_config": compression_config,
        "rand_seed": rand_seed
    }
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
    

    # Model Loading
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        path=model2path[model_name],
        device=device,
        cache_dir=CACHE_DIR
    )

    # Patching
    patch_models(
        compression_method=args.mode,
        model=model,
    )
    # config compression
    config_path = args.config_path
    if config_path is not None:
        compression_config = json.load(open(config_path, "r"))
        model = config_compress(
            model=model,
            method=args.mode,
            compression_config=compression_config,
            args=args
        )
    for dataset in datasets:
        data = load_dataset("THUDM/LongBench", f"{dataset}", split='test', trust_remote_code=True, cache_dir="./longbench_dataset")
        out_path = f"{out_path_prefix}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        if "pangu-7b-think" in model_name.lower():
            max_gen = 32768
            print("[INFO] Using thinking model, set max_gen_length to 32768")
        data_all = [data_sample for data_sample in data]
        # TODO: hard code single process, which use all gpus
        get_pred(model, tokenizer, data_all, max_length, max_gen, prompt_format, dataset, device, model_name, out_path)
