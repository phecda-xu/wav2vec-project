#! /usr/bin/bash

cd ../../.. > /dev/null

python fairseq/examples/wav2vec/wav2vec_manifest.py /home/aipc/data/aishell_1 \
--dest Work/wav2vec/data \
--ext wav \
--valid-percent 0.1








