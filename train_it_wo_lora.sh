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
SAVE_NAME=sft_v0.1_iter50000_lora_fix_v1_full_finetune

SAVE_PATH="./output/sft/${SAVE_NAME}/"
TENSORBOARD_PATH="./tensorboard/sft/${SAVE_NAME}/"

max_length=1024
micro_batch_size=4
global_batch_size=256

# train_iters = total_data * train_epochs // global_batch_size
# 361481 * 3 / 256 = 4236
train_epochs=3
train_iters=4236

# lr_warmup_iters=`expr $train_iters \/ 100`
lr_warmup_iters=50
lr_decay_iters=`expr $train_iters - $lr_warmup_iters`

eval_iter=50
eval_interval=50
save_interval=250


mkdir -p ${SAVE_PATH}
mkdir -p ${TENSORBOARD_PATH}

options=" \
	--max-completion-length 0 \
	--seq-length ${max_length} \
	--tokenizer-type LLaMATokenizer \
	--micro-batch-size ${micro_batch_size} \
	--train-iters ${train_iters} \
    --train-epochs ${train_epochs} \
	--lr-decay-iters ${lr_decay_iters} \
	--lr-warmup-iters ${lr_warmup_iters} \
	--lr 2e-5 \
	--min-lr 1e-6 \
	--lr-decay-style cosine \
	--log-interval 1 \
	--eval-iters ${eval_iter} \
	--eval-interval ${eval_interval} \
	--vocab-file tokenizer.model \
    --save-interval ${save_interval} \
	--save ${SAVE_PATH} \
	--tensorboard-dir ${TENSORBOARD_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--no-gradient-accumulation-fusion \
	--dataloader-type xgpt3 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 32 \
	--init-method-std 0.01 \
    --flash-attn \
	--bf16"

multimodal_options=" \
	--mm-config configs/instruction_tuning/v0.yaml \
	--use-learnable-tokens \
	--num-learnable-tokens 64 \
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 