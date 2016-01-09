import sys
import os
import time
import numpy as np

from chainer import functions as F
from chainer import serializers as S
from chainer import cuda

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm_trainer import LSTMTrainer


class VPPCTrainer(LSTMTrainer):
    def __init__(self, n_hidden, n_epoch=4000,
                 environment_size=(9, 9), sequence_length=100, gpu=-1):
        super(VPPCTrainer, self).__init__(n_hidden, n_epoch,
                                          environment_size=environment_size,
                                          sequence_length=sequence_length,
                                          gpu=gpu)

        self.n_output = 60

        self.setup_model()
        self.setup_gpu()
        self.setup_dataset()

    def loss(self, y, t):
        sigmoid_y = 1 / (1 + self.mod.exp(-y.data))
        mean_squared_error = ((t.data - sigmoid_y) ** 2).sum() / t.data.size
        return F.sigmoid_cross_entropy(y, t), mean_squared_error

    def evaluate(self, data, test=False):
        model = self.model.copy()
        model.reset_state()

        sum_error = 0
        for i in range(len(data['input'])):
            x = self.toVariable(data['input'][i], dtype='float32', train=False)
            t = self.toVariable(data['output'][i], dtype='int32', train=False)
            h, y = model(x)
            loss, mean_squared_error = self.loss(y, t)
            sum_error += mean_squared_error

        return sum_error / len(data['input'])

    def generate_data(self):
        sequence = self.dataset_generator.generate_sequence(
            self.sequence_length, self.offset_timing)
        return {
            'input':
                [a + b for a, b in
                 zip(sequence['action_units'], sequence['image_units'])],
            'output': sequence['image_units'][1:]}

    def train(self):
        cur_log_perp = self.mod.zeros(())
        accum_loss = 0
        print('[train]\ngoing to train %d epochs' % self.n_epoch)

        train_errors = []
        valid_errors_mean = []
        valid_errors_se = []

        for epoch in range(self.n_epoch):
            train_data = self.generate_data()

            for i in range(self.sequence_length):
                x = self.toVariable(train_data['input'][i], dtype='float32')
                t = self.toVariable(train_data['output'][i], dtype='int32')
                h, y = self.model(x)
                loss_i, mean_squared_error = self.loss(y, t)
                accum_loss += loss_i
                cur_log_perp += loss_i.data

                # truncated BPTT
                if (i + 1) % self.backprop_length == 0:
                    self.model.zerograds()
                    accum_loss.backward()
                    accum_loss.unchain_backward()  # truncate
                    accum_loss = 0
                    self.optimizer.update()

            if (epoch + 1) % self.validation_timing == 0:
                now = time.time()
                throughput = self.validation_timing / float(now - prev) \
                    if 'prev' in vars() else 0

                train_perp = self.evaluate(train_data)
                train_errors.append(train_perp)

                valid_perps = self.mod.zeros(self.validation_dataset_length)
                for i in range(self.validation_dataset_length):
                    valid_perps[i] = self.evaluate(self.validation_dataset[i])
                valid_perp_mean = np.mean(valid_perps, axis=0)
                valid_errors_mean.append(valid_perp_mean)
                valid_perp_se = np.std(valid_perps, axis=0) / \
                    np.sqrt(self.validation_timing)
                valid_errors_se.append(valid_perp_se)

                if epoch == 0:
                    perp = 0
                else:
                    perp = cuda.to_cpu(cur_log_perp) / self.validation_timing
                    perp = int(perp * 100) / 100.0

                print(
                    ('epoch: %d, train perp: %d, train mse: %.5f,'
                    + 'validation mse: %.5f (%.2f epochs/sec)') %
                    (epoch + 1, perp, train_perp, valid_perp_mean, throughput))

                if epoch >= 500:
                    self.optimizer.lr /= 1.2
                    print('learning rate: %.3f' % self.optimizer.lr)

                cur_log_perp = self.mod.zeros(())
                prev = now

                S.save_hdf5('vppc_lstm_%d.pkl' % self.n_hidden, self.model)

            sys.stdout.flush()

trainer = VPPCTrainer(60)
trainer.train()
