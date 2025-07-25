#!/bin/bash

# 使用单卡训练
CUDA_VISIBLE_DEVICES=0 python unlearn_harm.py \
    --model_name=LLM-Research/Meta-Llama-3-8B \
    --model_save_dir=models/Llama-3-8b_unlearned \
    --log_file=logs/Llama-3-8b-unlearn.log \
    --batch_size=2 \
    --use_lora \
