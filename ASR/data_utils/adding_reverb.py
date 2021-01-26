# coding:utf-8
#
# Date : 2020.03.21
# Author: phecda-xu < >
#
# DEC:
#     add reverb
import os
import sys
sys.path.append('ASR/data_utils/reverb')
sys.path.append('../ASR/data_utils/reverb')

import math
import random
import numpy as np
import pyrirgen as RG
from scipy import signal


class ReverbAugmentor(object):
    def __init__(self,
                 min_distance,
                 max_distance):
        self.c = 340  # Sound velocity (m/s)
        self.sr = 16000  # Sample rate (samples/s)
        self.min_distance = int(min_distance)
        self.max_distance = int(max_distance)

    def h_filter(self, T60):
        rp = 1  # Receiver position
        sp = random.randint(self.min_distance + 1, self.max_distance + 1)  # Source position
        r = [2, rp, 2]  # Receiver position [x y z] (m)
        s = [2, sp, 2]  # Source position [x y z] (m)
        L = [5, 4, 6]  # Room dimensions [x y z] (m)
        rt = round(random.uniform(0.2, 0.8), 1)  # Reflections Coefficients
        n = int(T60 * self.sr)  # Number of samples
        mtype = 'omnidirectional'  # Type of microphone 默认 omnidirectional 全方向的
        order = 1  # Reflection order
        dim = 3  # Room dimension
        ori = round(random.uniform(0, 2 * math.pi), 2)
        orientation = [ori, 0]  # Microphone orientation (rad)
        hp_filter = 1  # Enable high-pass filter
        h = RG.rir_generator(self.c, self.sr, s, r, L, reverbTime=rt, nSamples=n, micType=mtype, nOrder=order, nDim=dim,
                             orientation=orientation, isHighPassFilter=hp_filter)
        return np.array(h)

    def add_reverb(self, sig):
        T60 = random.uniform(0.4, 1.2)
        h = self.h_filter(T60)
        reverb_sig = signal.convolve(sig, h)
        reverb_sig = reverb_sig / max(h)
        return reverb_sig
