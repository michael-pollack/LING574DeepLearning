#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env

python run.py
python run.py --l2 0.00001 
python run.py --l2 0.00001 --word_dropout 0.3
