#!/bin/bash

# 使用单卡训练
CUDA_VISIBLE_DEVICES=0 python unlearn_harm.py \
    --model_name=LLM-Research/llama-2-7b \
    --model_save_dir=models/Llama-2-7b_unlearned \
    --log_file=logs/Llama-2-7b-unlearn.log \
    --batch_size=1 \
    --use_lora
