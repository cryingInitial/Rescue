#/bin/bash

# CIL CONFIG
NOTE="twf_just_run" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="twf"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=10
REPEAT=1
INIT_CLS=100
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1
    LAMBDA_FP_REPLAY=0.1
    LAMBDA_FP=0.005
    LAMBDA_DIVERSE_LOSS=0.1
    DER_ALPHA=0.3
    DER_BETA=0.9
    SAMPLES_PER_TASK=10000
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

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

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=2 python main_new.py --mode $MODE \
    --dataset $DATASET \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS\
    --rnd_seed $RND_SEED --samples_per_task $SAMPLES_PER_TASK \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --lambda_fp_replay $LAMBDA_FP_REPLAY --lambda_diverse_loss $LAMBDA_DIVERSE_LOSS \
    --lambda_fp $LAMBDA_FP --der_alpha $DER_ALPHA --der_beta $DER_BETA \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP 
done
