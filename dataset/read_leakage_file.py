# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:38
# @Author  : MA Ziqing
# @FileName: read_leakage_file.py

from scipy.io import wavfile
import numpy as np


class WavReader(object):
    def __init__(self):
        pass

    def read_all_wave_files(self):
        pass

    def read_one_wave_file(self):
        sample_rate, signal = wavfile.read('test.wav')
        return signal

