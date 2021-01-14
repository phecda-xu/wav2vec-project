#! /usr/bin/bash

export HYDRA_FULL_ERROR=1

fairseq-hydra-train \
    task.data=/home/aipc/Documents/wav2vec/data \
    --config-dir /home/aipc/Documents/wav2vec/V2/config \
    --config-name wav2vec2_base