#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env

# put your command for running word2vec.py here

python /dropbox/23-24/574/hw2/analysis.py --save_vectors vectors.tsv --save_plot vectors.png
