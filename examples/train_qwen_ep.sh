#!/bin/bash
# -------------------------------------------------------------------------
# Qwen2.5-32B-Instruct FSDP2 + EP Training Example
# -------------------------------------------------------------------------

# Environment Variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_DISABLED=true

# Distributed Configuration
NNODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT="12345"
NPROC_PER_NODE=8

# Parallelism Configuration
# Total GPUs = 8
# EP Size = 4  => 4 expert groups
# EFSDP Size = 8 / 4 = 2 => Each expert group has 2 GPUs doing FSDP for experts
EP_SIZE=4

# Training Arguments
MODEL_NAME="/home/wjq/ep_lf/Qwen3-30B-A3B"
OUTPUT_DIR="saves/qwen3-30b/ep_test"

# Run Command
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/llamafactory/v1/launcher.py \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL_NAME \
    --dataset identity \
    --template qwen \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 5 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --ep_size $EP_SIZE \
    --ep_model_adapter qwen2_moe \
    --dist_backend nccl \
    --enable_fsdp2 \
    --mixed_precision bf16 \
    --report_to none
