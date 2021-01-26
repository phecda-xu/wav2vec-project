#! /usr/bin/bash

# step 1: install virtualenv

info=`pip show virtualenv`

if [ -z "$info" ]
then
  pip install virtualenv -i https://mirrors.aliyun.com/pypi/simple/
fi

# step 2: venv
virtualenv -p python3.7 venv
source venv/bin/activate

# step 3: requirements
pip install -r requirements.txt

# step 4: fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

cd ..

# step 5: apex
if [ ! -d "apex/" ];then
  git clone --recursive https://github.com/NVIDIA/apex.git
fi

cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" --global-option="--xentropy"  --global-option="--fast_multihead_attn" ./

cd ..

# step 6: ctcdecode
if [ ! -d "ctcdecode/" ];then
  git clone --recursive https://github.com/parlance/ctcdecode.git
  cd ctcdecode/third_party/
else
  cd ctcdecode/third_party/
fi

if [ ! -f "openfst-1.6.7.tar.gz" ];then
  wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.7.tar.gz
fi

if [ ! -f "boost_1_67_0.tar.gz" ];then
  wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz
fi

tar -xzvf openfst-1.6.7.tar.gz
tar -xzvf boost_1_67_0.tar.gz

cd ..
pip install .
cd ..
mv ctcdecode/ ../


# step 7: rir
git clone https://github.com/phecda-xu/RIR-Generator.git
cd RIR-Generator

pip install -r requirements.txt
make
cd ..

mkdir ASR/data_utils/reverb
cp RIR-Generator/*.so ASR/data_utils/reverb

mv RIR-Generator ../