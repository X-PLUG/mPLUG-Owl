#!/bin/bash
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

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

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

EXP_NAME=sft_v0.1
SAVE_NAME=sft_v0.1_ft_grad_ckpt

SAVE_PATH="./output/${SAVE_NAME}/"

max_length=2048
micro_batch_size=1
global_batch_size=256
gradient_accumulation_steps=4

# train_iters = total_data * train_epochs // global_batch_size
# 361481 * 3 / 256 = 4236
train_epochs=3
train_iters=4236

lr_warmup_iters=50
lr_decay_iters=`expr $train_iters - $lr_warmup_iters`

eval_iter=50
eval_interval=50
save_interval=500

mkdir -p ${SAVE_PATH}

options=" \
	--pretrained-ckpt MAGAer13/mplug-owl-llama-7b-pt \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--num-training-steps ${train_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr 1e-5 \
	--min-lr 1e-6 \
	--eval-iters ${eval_iter} \
    --save-interval ${save_interval} \
	--save-path ${SAVE_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 32 \
	--gradient-checkpointing \
	--bf16"

multimodal_options=" \
	--mm-config configs/v0.yaml 
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pipeline/train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 