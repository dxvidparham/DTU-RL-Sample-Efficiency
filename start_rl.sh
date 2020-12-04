#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J ddpg
#BSUB -n 1
#BSUB -W 23:45
#BSUB -R "rusage[mem=16GB]"
#BSUB -u hello@nicklashansen.com
#BSUB -o %J.out
#BSUB -e %J.err

# Load modules
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

# Edit environment variables
unset PYTHONHOME
unset PYTHONPATH
export MUJOCO_GL=egl
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/zhome/88/6/108175/.mujoco/mujoco200/bin
export IS_BSUB_EGL=1

python3 src/main.py \
    --domain_name walker \
    --task_name walk \
    --action_repeat 4 \
    --mode baseline \
    --soda_augs conv \
    --exp_suffix default \
    --note color_hard \
    --seed 10 \
    --save_buffer \
    --save_freq 10000

# python3 src/train.py \
#     --domain_name gen3 \
#     --task_name reach \
#     --episode_length 50 \
#     --action_repeat 1 \
#     --frame_stack 1 \
#     --train_steps 100k \
#     --save_freq 100000 \
#     --eval_freq 2000 \
#     --mode soda \
#     --soda_augs conv \
#     --exp_suffix conv+bn+b256 \
#     --seed

# python3 src/train.py \
#     --domain_name gen3 \
#     --task_name push \
#     --episode_length 50 \
#     --action_repeat 1 \
#     --frame_stack 1 \
#     --mode dr \
#     --soda_augs conv \
#     --exp_suffix default \
#     --seed 3 \
#     --resume \
#     --resume_step 490000 \
#     --save_freq 10000 \
#     --save_buffer