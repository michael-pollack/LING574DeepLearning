#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env

# put your command for running word2vec.py here

python word2vec.py --num_epochs 6 --embedding_dim 15 --learning_rate 0.2 --min_freq 5 --num_negatives 15 
