#! /usr/bin/bash

cd ../../.. > /dev/null

export HYDRA_FULL_ERROR=1

fairseq-hydra-train \
    task.data=Work/wav2vec/data \
    --config-dir Work/wav2vec/V2/config \
    --config-name wav2vec2_base