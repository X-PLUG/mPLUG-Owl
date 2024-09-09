#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
###############################################################################
# SERVER_PORT=$1
DIR=`pwd`

EXP_NAME="Inference"
echo $EXP_NAME

gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1)

if [[ $gpu_model == *"A100"* ]]; then
    b=4
    DTYPE="--bf16"
elif [[ $gpu_model == *"V100"* ]]; then
    b=4
    DTYPE="--fp16"
else
    b=4
    DTYPE="--bf16"
fi
mp=1

GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# GPUS_PER_NODE=1
# Change for multinode config
if [ $MASTER_ADDR ];then
	echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
fi

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
echo $GPUS_PER_NODE
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# launch_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS"
launch_cmd="torchrun $DISTRIBUTED_ARGS"
NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
MAX_POSITION_EMBEDDINGS=2048
SEQ_LEN=2048

NUM_WORKERS=16

if [[ -z "$IMG_H" ]]
then
    IMG_H=224
else
    b=$((b / 2))
fi
b=1
program_cmd="${DIR}/tasks/general_task_pipeline2.py \
       --load $CHECKPOINT_PATH "

evaluation_options=" \
    --evaluation-plan $EVAL_PLAN \
    "

echo $launch_cmd $program_cmd $evaluation_options # $multimodal_options

$launch_cmd $program_cmd $evaluation_options # $multimodal_options
