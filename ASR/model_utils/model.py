# coding: utf-8
#
#
#
#
#
import math
import torch
import fairseq
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def load_wav2vec_model(model_path):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = model[0]
    model.eval()
    return model


class TDNLayer(nn.Module):
    def __init__(self, context: int, stride: int, input_channels: int, output_channels: int, dilate: int, pad: list):
        super(TDNLayer, self).__init__()
        self.input_dim = input_channels
        self.output_dim = output_channels
        self.pad = pad
        self.temporal_conv = weight_norm(
            nn.Conv1d(input_channels,
                      output_channels,
                      kernel_size=context,
                      stride=stride,
                      dilation=dilate))
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.activate = nn.Hardtanh(0, 20, inplace=True)

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_channels, sequence_length]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, output_dim, len(valid_steps)]
        """
        x = F.pad(x, pad=self.pad, mode='replicate')
        x = self.temporal_conv(x)
        x = self.activate(x)
        return x

    @staticmethod
    def check_valid_context(x, context: int, dilate: int) -> None:
        """
        Check whether the context and dilate is suitable with input x size
        context and dilate is the conv params
        :param x: input array
        :param context: context size
        :param dilate: dilate coefficient.
        """
        context_list = [i for i in range(0, context * dilate, dilate)]
        context_list = [i - context_list[context // 2] for i in context_list]
        index_size = context_list[-1] - context_list[0] + 1
        if x.size()[1] < index_size:
            raise ValueError("input feature dim should larger than reception filed, which is:{}|{}".format(
                x.size()[1], index_size
            ))


class TDNNet(nn.Module):
    def __init__(self, labels, audio_conf=None, input_channels=512):
        super(TDNNet, self).__init__()
        self.audio_conf = audio_conf
        self.labels = labels
        num_classes = len(self.labels)  # space index
        net_arch = self.audio_conf["net_arch"]
        if net_arch == "super":
            self.tdnn = nn.Sequential(
                TDNLayer(context=3, stride=1, input_channels=input_channels, output_channels=2000, dilate=1, pad=[13, 9]),
                TDNLayer(context=3, stride=1, input_channels=2000, output_channels=800, dilate=2, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=800, output_channels=800, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=800, output_channels=800, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=800, output_channels=800, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=800, output_channels=800, dilate=2, pad=[0, 0]),
                TDNLayer(context=6, stride=1, input_channels=800, output_channels=2000, dilate=1, pad=[0, 0]),
            )
            self.fc = nn.Linear(2000, num_classes, bias=False)
        elif net_arch == "large":
            self.tdnn = nn.Sequential(
                TDNLayer(context=3, stride=1, input_channels=input_channels, output_channels=1024, dilate=1, pad=[13, 9]),
                TDNLayer(context=3, stride=1, input_channels=1024, output_channels=512, dilate=2, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=512, output_channels=256, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=256, output_channels=128, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=128, output_channels=256, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=256, output_channels=512, dilate=2, pad=[0, 0]),
                TDNLayer(context=6, stride=1, input_channels=512, output_channels=1024, dilate=1, pad=[0, 0]),
            )
            self.fc = nn.Linear(1024, num_classes, bias=False)
        elif net_arch == "medium":
            self.tdnn = nn.Sequential(
                TDNLayer(context=3, stride=1, input_channels=input_channels, output_channels=128, dilate=1, pad=[13, 9]),
                TDNLayer(context=3, stride=1, input_channels=128, output_channels=64, dilate=2, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=64, output_channels=40, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=40, output_channels=40, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=40, output_channels=40, dilate=3, pad=[0, 0]),
                TDNLayer(context=2, stride=1, input_channels=40, output_channels=64, dilate=2, pad=[0, 0]),
                TDNLayer(context=6, stride=1, input_channels=64, output_channels=128, dilate=1, pad=[0, 0]),
            )
            self.fc = nn.Linear(128, num_classes, bias=False)
        elif net_arch == "small":
            self.tdnn = nn.Sequential(
                TDNLayer(context=3, stride=1, input_channels=input_channels, output_channels=64, dilate=1, pad=[13, 9]),
                TDNLayer(context=3, stride=1, input_channels=64, output_channels=40, dilate=3, pad=[0, 0]),
                TDNLayer(context=15, stride=1, input_channels=40, output_channels=40, dilate=1, pad=[0, 0]),
            )
            self.fc = nn.Linear(40, num_classes, bias=False)
        else:
            raise ValueError("wrong type of args.net_arch : {}, only support small or large".format(net_arch))

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        x = x.squeeze(1)
        embdding = self.tdnn(x)
        x = embdding.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        output_lengths = self.get_seq_lens_2(lengths)
        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        return x, output_lengths, embdding

    def get_seq_lens_2(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.tdnn.modules():
            if type(m) == TDNLayer:
                padding = m.pad
            if type(m) == nn.modules.conv.Conv1d:
                seq_len = ((seq_len + padding[0] + padding[1] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) // m.stride[0] + 1)
        return seq_len.int()

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = TDNNet.load_model_package(package)
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(
                    labels=package['labels'],
                    audio_conf=package['audio_conf'])
        model.load_state_dict(package['state_dict'])
        return model

    def serialize_state(self):
        return {
            'audio_conf': self.audio_conf,
            'labels': self.labels,
            'state_dict': self.state_dict()
        }

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


class FCFintune(nn.Module):
    def __init__(self, labels, audio_conf=None, input_channels=512):
        super(FCFintune, self).__init__()
        self.audio_conf = audio_conf
        self.labels = labels
        num_classes = len(self.labels)  # space index
        self.fc = nn.Linear(input_channels, num_classes, bias=False)

    def forward(self, x, lengths):
        embdding = x
        x = embdding.transpose(1, 2).transpose(0, 1).contiguous()
        lengths = lengths.cpu().int()
        x = self.fc(x)
        x = x.transpose(0, 1)
        return x, lengths, embdding

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = FCFintune.load_model_package(package)
        return model

    @classmethod
    def load_model_package(cls, package):
        model = cls(
                    labels=package['labels'],
                    audio_conf=package['audio_conf'])
        model.load_state_dict(package['state_dict'])
        return model

    def serialize_state(self):
        return {
            'audio_conf': self.audio_conf,
            'labels': self.labels,
            'state_dict': self.state_dict()
        }

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
