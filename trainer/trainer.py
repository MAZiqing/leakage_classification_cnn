# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:21
# @Author  : MA Ziqing
# @FileName: trainer.py

import torch
from torch.utils.data import DataLoader
from models.leakage_tcn import LeakageTCN
from dataset.leak_dataset import LeakDataset
import datetime


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = LeakageTCN(input_size=args.input_channel,
                                output_size=args.output_size,
                                num_channels=[args.nhid] * args.levels,
                                kernel_size=args.ksize,
                                dropout=args.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr)
        self.dataset = LeakDataset(signal_length_maximum=args.signal_length_maximum)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def train_all(self):
        for epoch in range(0, self.args.epochs):
            loss = self.train()
            print('time={} | epoch={} | loss={} |'.format(datetime.datetime.now(), epoch, loss))

    def train(self):
        self.model.train()
        total_loss = 0.0
        i = 1
        for i, (inputs, labels) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # torch.Size([64, 10, 9])
            loss = self.cross_entropy_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.cpu().detach().numpy()
        return total_loss / i
