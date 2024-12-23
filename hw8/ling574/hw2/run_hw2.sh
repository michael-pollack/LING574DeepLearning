#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dropbox/23-24/574/env

# put your command for running word2vec.py here
python word2vec.py --num_epochs 15 --save_vectors vectors.tsv --embedding_dim 15 --learning_rate 0.2 --min_freq 5 --num_negatives 15


python /mnt/dropbox/23-24/574/hw2/analysis.py --save_vectors vectors.tsv --save_plot vectors.png
