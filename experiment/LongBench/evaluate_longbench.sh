#!/usr/bin/env bash

model_name="pangu-1b-v11" # pangu-7b-v11, pangu-7b-think-v11

method="entropy_infer"

config_path="config/entropy_infer_config/entropy_infer.json"

python experiment/LongBench/pred.py \
    --model_name=$model_name \
    --config_path=$config_path \
    --mode=$method