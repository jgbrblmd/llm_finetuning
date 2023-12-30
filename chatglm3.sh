#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,5
export WANDB_DISABLED=true
export OMP_NUM_THREADS=4

torchrun --nnodes 1 --nproc-per-node 4 --standalone chatglm3_ft.py --base_model /nvme/opt/LLM/ZhipuAI/chatglm3-6b --data_path data/train.jsonl --mode 4 
