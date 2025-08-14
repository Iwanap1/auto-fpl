#!/bin/bash
#PBS -lselect=1:ncpus=8:mem=32gb
#PBS -lwalltime=24:00:0

set -euo pipefail
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate auto_fpl
cd $HOME/auto-fpl/iwans_system
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python rllib_train.py