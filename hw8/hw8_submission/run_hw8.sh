#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env

python run.py --train_source /dropbox/23-24/574/data/europarl-v7-es-en/train.en.txt --train_target /dropbox/23-24/574/data/europarl-v7-es-en/train.es.txt --output_file test.en.txt.es --num_epochs 8 --embedding_dim 16 --hidden_dim 64 --num_layers 2

python chrF++.py -nw 0 -R /dropbox/23-24/574/data/europarl-v7-es-en/test.es.txt -H test.en.txt.es > test.en.txt.es.score
