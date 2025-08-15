#!/bin/bash
#PBS -lselect=1:ncpus=16:mem=128gb
#PBS -lwalltime=72:00:0

set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate auto_fpl
cd $HOME/auto-fpl/discrete_sys
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1
python train_masked_ppo.py