#! /usr/bin/bash

# venv
virtualenv -p python3.7 venv
source venv/bin/activate

# requirements
pip install -r requirements.txt

# fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

cd ..

#apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" --global-option="--xentropy"  --global-option="--fast_multihead_attn" ./

cd ..