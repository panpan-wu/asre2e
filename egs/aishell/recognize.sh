#!/bin/bash

download_dir="/home/pan/DirsShared/data/asr/aishell_test"

# python aishell.py --download_dir ${download_dir}

asre2e_recognize \
    --config conf/train.yaml \
    --test_data data/test/data_info.csv \
    --char_map data/char_map.txt \
    --checkpoint exp/checkpoints/checkpoint.1.pt \
    --search_type ctc_beam_search \
    --beam_size 4
