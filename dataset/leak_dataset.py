# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:38
# @Author  : MA Ziqing
# @FileName: leak_dataset.py

import os
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset
from trainer.trainer import *
from dataset.read_leakage_file import WavReader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LeakDataset(Dataset):
    def __init__(self, signal_length_maximum):
        '''
        :param dataset_type: str, dataset type
        :param encoder_sequence_length: int, the length of encoder
        :param decoder_sequence_length: int, the length of decoder
        :param target_sensor: str, the id of target sensor
        '''
        self.signal_length_maximum = signal_length_maximum
        wav_reader = WavReader()
        result_list = wav_reader.read_all_wave_files()
        self.input_signal = torch.tensor([i[0][:self.signal_length_maximum] for i in result_list],
                                         dtype=torch.float32).transpose(1, 2).to(device)
        self.output_label = torch.tensor([i[1] for i in result_list], dtype=torch.long).to(device)
        print('dataset prepared !')

    def __len__(self):
        return self.output_label.shape[0]

    def __getitem__(self, idx):
        inputs = self.input_signal[idx]
        outputs = self.output_label[idx]
        return inputs, outputs


def test():
    dataset = LeakDataset(1000)
    sample = dataset.__getitem__(1)
    print(sample)


if __name__ == '__main__':
    test()
