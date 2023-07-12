#/bin/bash

# CIL CONFIG
NOTE="0706_ocs_.8" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ocs"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=0
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
# SEEDS="1 2 3"
SEEDS="1"
N_TASKS=5

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    SAMPLES_PER_TASK=10000
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=40; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=1
    SAMPLES_PER_TASK=10000
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=40; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

rm -f nohup.out

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=9 python main_new.py --mode $MODE \
    --dataset $DATASET \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS\
    --rnd_seed $RND_SEED --samples_per_task $SAMPLES_PER_TASK \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --n_tasks $N_TASKS\
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP
done
