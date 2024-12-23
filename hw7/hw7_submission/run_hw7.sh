#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env

python run.py
python run.py --num_epochs 60
