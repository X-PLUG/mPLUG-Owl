apt update
apt install -y default-jdk
# fuser -k /dev/nvidia*
nvidia-smi

export PYTHONPATH='./'
export SDPA=1

export IMG_H=384
export EXP_NAME=auto
export MODEL_CONFIG=auto
export CHECKPOINT_PATH=../mPLUG/mPLUG-Owl3-7B-240728

EVAL_PLAN=tasks/plans/all.yaml bash tasks/general_task_pipeline2.sh
