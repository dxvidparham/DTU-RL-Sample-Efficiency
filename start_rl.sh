#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J sac_impl
#BSUB -n 1
#BSUB -W 23:45
#BSUB -R "rusage[mem=16GB]"
#BSUB -u markus.boebel@tum.de
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

# Load modules
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8
module load ffmpeg/4.2.2

# Edit environment variables
#unset PYTHONHOME
#unset PYTHONPATH
#export MUJOCO_GL="egl"
#export PATH="$HOME/.local/bin:$PATH"
#export IS_BSUB_EGL=1

python3 main.py --env-domain=cartpole\
                --env-task=swingup\
                --seed=1\
                --save_video\
                --recording_interval=200\
                --episodes=50000\
                --max_eval=30\
                --log_level="INFO"\
                --gpu_device=0
