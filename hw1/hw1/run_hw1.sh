#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
# if you install miniconda in a different directory, try the following command
# source path_to_anaconda3/miniconda3/etc/profile.d/conda.sh
# if you install the full anaconda package instead of just miniconda, try:
# source ~/anaconda3/etc/profile.d/conda.sh

conda activate /dropbox/23-24/574/env/
python main.py --text_file /dropbox/23-24/574/data/sst/train-reviews.txt --output_file train_vocab_base.txt
python main.py --text_file /dropbox/23-24/574/data/sst/train-reviews.txt --output_file train_vocab_freq5.txt --min_freq 5

# include your commands here 
