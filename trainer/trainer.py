# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 16:21
# @Author  : MA Ziqing
# @FileName: trainer.py

from models.leakage_tcn import LeakageTCN


class Trainer(object):
    def __init__(self):
        self.batch_size = 32
        self.epoch = 100
        self.model = LeakageTCN()
        self.optimizer = None

    def train(self):
        self.model.train()
        total_loss = 0.0
        total_loss_l1 = 0.0
        i = 1
        for i, (input_p, label_p, datetime_i) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            if input_p.shape[0] == args.batch_size:
                outputs_p = self.model(input_p, label_p)  # torch.Size([64, 10, 9])
                label_p = label_p[:, -args.decoder_sequence_length:]
                # outputs_p = outputs_p.squeeze()
                if i == 100:
                    if self.epoch % 10 == 0:
                        print('label:', np.around(label_p[0].cpu().detach().numpy(), decimals=5))
                        print('output:', np.around(outputs_p[0].cpu().detach().numpy(), decimals=5))
                loss = self.mse_loss(outputs_p, label_p)
                l1loss = self.l1_loss(outputs_p, label_p)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().detach().numpy()
                total_loss_l1 += l1loss.cpu().detach().numpy()
        self.train_mse_loss.append(total_loss / i)
        self.result_df['train_mse_curve'] = [self.train_mse_loss]
        return total_loss / i, total_loss_l1 / i
