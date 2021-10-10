#!/bin/bash

d=model_params_from_wenet
mkdir $d
cd $d
wget http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_conformer_exp.tar.gz
tar -xzvf 20210204_conformer_exp.tar.gz
rm -rf 20210204_conformer_exp.tar.gz
