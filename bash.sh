#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,3
export WANDB_DISABLED=true
export OMP_NUM_THREADS=32

#torchrun --nnodes 1 --nproc-per-node 4 --standalone chatglm3_ft.py --base_model /nvme/opt/LLM/ZhipuAI/chatglm3-6b --data_path data/train.jsonl --mode 4 
#torchrun --nnodes 1 --nproc-per-node 2 --standalone chatglm3_ft.py --base_model /nvme/opt/LLM/Yi-34B-200K-Llamafied-AWQ --data_path data/train.jsonl --mode 4 
#torchrun --nnodes 1 --nproc-per-node 2 --standalone chatglm3_ft.py --base_model /nvme/opt/LLM/SOLAR-10.7B-Instruct-v1.0-4.0bpw-h6-exl2-2 --data_path data/train.jsonl --mode exl
torchrun --nnodes 1 --nproc-per-node 2 --standalone finetune.py --base_model /nvme/opt/LLM/SOLAR-10.7B-Instruct-v1.0-GPTQ --data_path data/alpaca_data.json --mode gptq