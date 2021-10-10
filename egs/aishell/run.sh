#!/bin/bash

download_dir="/home/pan/DirsShared/data/asr/aishell_test"

# python aishell.py --download_dir ${download_dir}

asre2e_train \
    --config conf/train.yaml \
    --train_data data/train/data_info.csv \
    --model_dir exp/checkpoints \
    --checkpoint exp/state_dict_from_wenet.pt \
    --global_cmvn exp/global_cmvn.json
