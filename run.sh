#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/20-21/575k/env/
python generate_embeddings.py $1 $2

