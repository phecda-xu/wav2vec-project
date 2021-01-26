# wav2vec-project

facebook开源的wav2vec代码使用学习记录

## 目录

- wav2vec 训练及推理
- 搭建ASR 微调

## 一、wav2vec 训练及推理

### 1.1 数据处理

```markdown
由于wav2vec的训练不需要标注信息，所以数据处理脚本比较简单，只需生产音频文件路劲以及音频时长的列表即可。
faieseq中提供了现成的脚本使用。wav2vec和wav2vec2的数据处理方法一样。
```

- [run_data.sh](Work/wav2vec/V1/run_data.sh)

```
需要设置的参数：
音频文件夹位置： /*/data/aishell_1
生成tsv文件保存地址：--dest Work/wav2vec/data
音频文件格式： --ext wav
验证集数据的占比： --valid-percent 0.1
```

- 生成数据列表： train.tsv/valid.tsv

```
data_aishell/wav/train/S0170/BAC009S0170W0378.wav	63102
data_aishell/wav/train/S0170/BAC009S0170W0184.wav	88336
data_aishell/wav/train/S0170/BAC009S0170W0285.wav	74336
data_aishell/wav/train/S0170/BAC009S0170W0259.wav	94959
data_aishell/wav/train/S0170/BAC009S0170W0191.wav	110015
```

### 1.2 训练模型

- V1:[ run_train.sh](Work/wav2vec/V1/run_train.sh)直接设置参数

```
需要修改的参数：
tsv文件地址：Work/wav2vec/data
模型保存地址：--save-dir models/v1
```

- V2：[run_train.sh](Work/wav2vec/V2/run_train.sh)通过[yaml配置文件](Work/wav2vec/V2/config/wav2vec2_base.yaml)设置

```
需要修改的参数：
tsv文件地址： task.data=Work/wav2vec/data \
yaml配置文件地址： --config-dir Work/wav2vec/V2/config \
yaml配置文件名称： --config-name wav2vec2_base
```

### 1.3 推理

- V1: [wav2vec_1.py](wav2vec_1.py)

```markdown
分两步进行，先将raw数据经过feature_extractor进行特征提取；
再经过上下文网络feature_aggregator进行上下文信息融合；
此外还可以得到CPC方法的输出；
详见 wav2vec_1.py 中示例。
```

- V2: [wav2vec_2.py](wav2vec_2.py)

```markdown
与V1不同，可以直接得到上下文信息输出；
同时还有更多的中间信息输出,需要注意参数设置 features_only=True；
详见 wav2vec_2.py 中示例。
```

## 二、搭建ASR微调

```markdown
目前能想到的微调方式有两种：
一种是原来的网络的权值与新增结构一起调整，
另一种是原网络的权值不变只调新增结构的权值。

由于原网络的改写比较复杂，暂不进行深入研究，所以选择第二种方式进行微调。

微调两种形式：
加全连接层 + CTC
加TDNN网络 + CTC
这里主要实现这两种
```

### 2.1 数据处理

```markdown
以aishell_1数据集为例说明数据处理过程，处理过程参考 paddlepaddle/DeepSpeech2 的实现
```

- [run_data.sh](Work/aishell_1/run_data.sh)

```markdown
先处理语音数据，生成manifest.train, manifest.test, manifest.dev文件;
再处理噪声数据, 生成manifest文件；
最后生成词典文件 vocab.txt
```

- [aishell_1.py](ASR/data_utils/aishell_1/aishell_1.py)

```markdown
处理语音数据脚本，生产如下格式数据：
{"audio_filepath": 
     "/*/*/data/aishell_1/data_aishell/wav/dev/S0724/BAC009S0724W0417.wav", 
 "duration": 5.8999375, 
 "text": "最终因是非停不了落选收场", 
 "pinyin": "z-zhui z-zhong yin-ing s-shi fei tin-ing bu niao nuo xuan s-shou c-chang", 
 "fully_pinyin": "zui4 zhong1 yin1 shi4 fei1 ting2 bu4 liao3 luo4 xuan3 shou1 chang3"
}
```

- [noise.py](ASR/data_utils/aishell_1/noise.py)

```markdown
生成噪声音频的manifest文件，格式与语音数据格式一致
```

### 2.2 训练

- [run_train.sh](Work/aishell_1/run_train.sh)

```markdown
参数设置说明：
配置加混响的库文件的环境变量，如果配置参数中没有 --reverberation，那么这一步有没有都不影响
path=`pwd`
export LD_LIBRARY_PATH=$path/ASR/data_utils/reverb:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH:"$LD_LIBRARY_PATH


选择wav2vec版本及对应的模型
--vector-model-arch V1 \
--vector-model-path models/V1/wav2vec_large.pt \
```

### 2.3 测试

- [run_test.sh](Work/aishell_1/run_test.sh)

```markdown
选择wav2vec版本及对应的模型
--vector-model-arch V1 \
--vector-model-path models/V1/wav2vec_large.pt \

选择测试的微调模型结构以及模型
--net-arch medium \
--model-path models/aishell_1/tdnn_final.pth
```

## 未完成

- 调试代码，解决bug
- 验证base的微调结果
- 验证TDNN的微调结果

## 参考链接

- [SeanNaren: deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) 
- [PaddlePaddle: DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)
- [pytorch: fairseq](https://github.com/pytorch/fairseq)