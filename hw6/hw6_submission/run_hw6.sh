#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env

python run.py
python run.py --l2 0.0004 --dropout 0.5
python run.py --lstm
python run.py --lstm --l2 0.0004 --dropout 0.5