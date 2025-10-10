#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate species

python3 celeb_preprocess.py $1 $2