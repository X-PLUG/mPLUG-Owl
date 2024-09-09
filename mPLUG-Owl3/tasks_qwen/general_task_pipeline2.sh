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
program_cmd="${DIR}/tasks_qwen/general_task_pipeline2.py \
       --load $CHECKPOINT_PATH "

    #    --use-cpu-initialization \
    #    --distributed-timeout-minutes 600 \
    #    --tensor-model-parallel-size $mp \
    #    --num-layers ${NUM_LAYERS} \
    #    --hidden-size ${HIDDEN_SIZE} \
    #    --num-attention-heads ${NUM_ATTN_HEADS} \
    #    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    #    --seq-length ${SEQ_LEN} \
    #    ${DTYPE} \
    #     --no-bias \
    #     --fp32-rope \
    #     --untie-embeddings-and-output-weights \
    #     --add-more-sp-tokens \
    #     --qkv-bias \
    #     --outer-rope \
    #     --hidden-dropout 0 \
    #     --attention-dropout 0 \
    #     --make-ffn-dim-multiple-of 256 \
    #     --make-vocab-size-divisible-by 1 \
    #     --activation swiglu \
    #     --use-rmsnorm \
    #     --pos-emb rotary \
    #     --seed 3407 \
    #     --micro-batch-size $b \
    #     --no-gradient-accumulation-fusion \
    #    --log-interval 1 \
    #    --merge-file ./qwen_15w.tiktoken \
    #    --vocab-file ./qwen_15w.tiktoken \
    #    --tokenizer-type QWenTokenizer \
    #    --load $CHECKPOINT_PATH "


# multimodal_options=" --mm-config ${EXP_NAME}" # \
    # --finetune \
    # --no-load-optim \
    # --use-qwen-vision \
    # --use-torch-gelu \
    # --freeze-vit \
    # --freeze-gpt \
    # --v-layernorm-epsilon 1e-6 \
    # --v-layer-decay 0.95 \
    # --img-h $IMG_H \
    # --img-w $IMG_H \
    # --patch-dim 14 \
    # --v-num-layers 48 \
	# --v-hidden-size 1664 \
    # --v-ffn-hidden-size 8192 \
	# --v-num-attention-heads 16 \
	# --use-learnable-tokens \
	# --num-learnable-tokens 64 \
    # --v2t-num-layers 6 \
    # --v-layernorm-type torch \
    # --add-v2t-pos-emb
    # "

# if [ ! -z "$MULTIWAY" ]; then
#     multimodal_options+=" --multiway $MULTIWAY"
# fi


evaluation_options=" \
    --evaluation-plan $EVAL_PLAN \
    "

echo $launch_cmd $program_cmd $evaluation_options # $multimodal_options

$launch_cmd $program_cmd $evaluation_options # $multimodal_options
