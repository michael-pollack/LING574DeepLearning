#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dropbox/23-24/574/env
python /dropbox/23-24/574/check_hw.py /dropbox/23-24/574/languages /dropbox/23-24/574/hw2/submit-file-list hw2.tar.gz
