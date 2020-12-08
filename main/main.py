# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:06
# @Author  : MA Ziqing
# @FileName: main.py.py

from trainer.trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size')
parser.add_argument('--input_channel', type=int, default=1,
                    help='input channel')
parser.add_argument('--output_size', type=int, default=2,
                    help='output size (number of classes)')
parser.add_argument('--signal_length_maximum', type=int, default=1000,
                    help='signal_length_maximum')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=2,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--nhid', type=int, default=16,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Nott',
                    help='the dataset to run (default: Nott)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

args = parser.parse_args()


def main():
    trainer = Trainer(args)
    trainer.train_all()


if __name__ == '__main__':
    main()

