#!/usr/bin/env bash

cd ../.. > /dev/null

CUDA_VISIBLE_DEVICES='' \
python -u test.py \
--cuda \
--half \
--verbose \
--checkpoint \
--reverberation \
--speed-volume-perturb \
--vector-model-arch V1 \
--vector-model-path models/V1/wav2vec_large.pt \
--test-manifest Work/aishell_1/data/manifest.aishell_1.test \
--noise-dir Work/noise/manifest.farfiled.background.train \
--noise-max 0.8 \
--noise-prob 0.8 \
--reverb-prob 0.8 \
--batch-size 4 \
--num-workers 2 \
--net-arch medium \
--word-form pinyin \
--lm-workers 4 \
--lm-path None \
--model-path models/aishell_1/tdnn_final.pth


# --vector-model-arch V1 or V2
# --net-arch base or small or medium or large or super
# --word-form pinyin or sinogram or english