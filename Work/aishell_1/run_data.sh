#! /usr/bin/env bash

cd ../.. > /dev/null

# download data, generate manifests
python ASR/data_utils/aishell_1/aishell_1.py \
--manifest_prefix='Work/aishell_1/data/manifest.aishell_1' \
--target_dir='~/data/aishell_1' \

if [ $? -ne 0 ]; then
    echo "Prepare Aishell failed. Terminated."
    exit 1
fi

# background data
python ASR/data_utils/aishell_1/noise.py \
--manifest_prefix='Work/noise/manifest.farfiled.background' \
--target_dir='~/data/_Farfiled_background_' \

if [ $? -ne 0 ]; then
    echo "Prepare Aishell failed. Terminated."
    exit 1
fi


# generate vocab.txt
python -u ASR/data_utils/build_vocab.py \
--manifest_paths 'Work/aishell_1/data/manifest.aishell_1.train' 'Work/aishell_1/data/manifest.aishell_1.dev' 'Work/aishell_1/data/manifest.aishell_1.test' \
--count_threshold=0 \
--vocab_path='Work/aishell_1/data/vocab.txt' \
--word_form='pinyin' \

if [ $? -ne 0 ]; then
    echo "Build vocabulary failed. Terminated."
    exit 1
fi


echo "Aishell data preparation done."
exit 0
