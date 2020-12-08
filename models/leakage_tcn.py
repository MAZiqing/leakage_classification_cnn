# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:25
# @Author  : MA Ziqing
# @FileName: leakage_tcn.py

import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet


class LeakageTCN(nn.Module):
    def __init__(self, input_size, output_size=2, num_channels=[1], kernel_size=32, dropout=0.5):
        super(LeakageTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)
