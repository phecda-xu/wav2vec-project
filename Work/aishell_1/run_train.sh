#!/usr/bin/env bash

cd ../.. > /dev/null

path=`pwd`
export LD_LIBRARY_PATH=$path/ASR/data_utils/reverb:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH:"$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES='' \
python -u train.py \
--opt-level O0 \
--reverberation \
--speed-volume-perturb \
--checkpoint \
--load-auto-checkpoint \
--checkpoint-per-iteration 10000 \
--vector-model-arch V1 \
--vector-model-path models/V1/wav2vec_large.pt \
--train-manifest Work/aishell_1/data/manifest.aishell_1.train \
--val-manifest Work/aishell_1/data/manifest.aishell_1.dev \
--noise-dir Work/noise/manifest.farfiled.background.train \
--max-durations 15.0 \
--min-durations 0.4 \
--noise-max 0.8 \
--noise-prob 0.8 \
--reverb-prob 0.8 \
--batch-size 4 \
--num-workers 2 \
--epochs 200 \
--net-arch medium \
--word-form pinyin \
--labels-path Work/aishell_1/data/vocab.txt \
--save-folder models/aishell_1/ \


# --vector-model-arch V1 or V2
# --net-arch small or medium or large or super
# --word-form pinyin or sinogram or english
#