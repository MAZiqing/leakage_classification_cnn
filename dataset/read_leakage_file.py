# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:38
# @Author  : MA Ziqing
# @FileName: read_leakage_file.py

import os
from scipy.io import wavfile
import numpy as np


class WavReader(object):
    def __init__(self):
        self.path = os.path.join(os.path.split(__file__)[0], 'wav_file_dataset')

    def read_all_wave_files(self):
        result_list = []
        classes = os.listdir(self.path)
        for one_class in classes:
            class_path = os.path.join(self.path, one_class)
            samples = os.listdir(class_path)
            for sample in samples:
                sample_path = os.path.join(class_path, sample)
                signal = self.read_one_wave_file(sample_path)
                result_list += [[signal, int(one_class)]]
        return result_list

    @staticmethod
    def read_one_wave_file(path):
        sample_rate, signal = wavfile.read(path)
        # signal 是一个 n*2 的 numpy array，2 表示有左右两个声道，我们随机选择了一个声道，后面可以再调整
        return signal[:, 0:1]


def test():
    wav_reader = WavReader()
    result_list = wav_reader.read_all_wave_files()
    print(result_list)


if __name__ == '__main__':
    test()
