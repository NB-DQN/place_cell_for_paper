import sys
import os
import time
import numpy as np

from chainer import functions as F
from chainer import serializers as S
from chainer import cuda

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm_trainer import LSTMTrainer


class PIPCTrainer(LSTMTrainer):
    def __init__(self, n_hidden, offset_timing=2, n_epoch=1000,
                 environment_size=(9, 9), sequence_length=100, gpu=-1):
        super(PIPCTrainer, self).__init__(n_hidden, n_epoch,
                                          environment_size=environment_size,
                                          sequence_length=sequence_length,
                                          offset_timing=offset_timing,
                                          gpu=gpu)

        self.n_output = environment_size[0] * environment_size[1]

        self.setup_model()
        self.setup_gpu()
        self.setup_dataset()

    def loss(self, y, t):
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def evaluate(self, data, test=False):
        model = self.model.copy()
        model.reset_state()

        sum_accuracy = 0
        for i in range(len(data['input'])):
            x = self.toVariable(data['input'][i], dtype='float32', train=False)
            t = self.toVariable(data['output'][i], dtype='int32', train=False)
            h, y = model(x)
            loss, accuracy = self.loss(y, t)
            sum_accuracy += accuracy.data

        error = 1 - sum_accuracy / len(data['input'])
        return error  # return error, not accuracy!!

    def generate_data(self, go_away_from_start=False):
        sequence = self.dataset_generator.generate_sequence(
            self.sequence_length, self.offset_timing, go_away_from_start)
        return {
            'input':
                [a + b for a, b in
                 zip(sequence['action_units'], sequence['image_units'])],
            'output': sequence['coordinate_ids'][1:]}

    def train(self):
        cur_log_perp = self.mod.zeros(())
        accum_loss = 0
        print('[train]\ngoing to train %d epochs' % self.n_epoch)

        for epoch in range(self.n_epoch):
            if epoch <= self.n_epoch / 2:
                train_data = self.generate_data()
            else:
                train_data = self.generate_data(go_away_from_start=True)

            for i in range(self.sequence_length):
                x = self.toVariable(train_data['input'][i], dtype='float32')
                t = self.toVariable(train_data['output'][i], dtype='int32')
                h, y = self.model(x)
                loss_i, accuracy_i = self.loss(y, t)
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

                train_perp, valid_perp_mean, valid_perp_se, perp = \
                    self.validate(epoch, train_data, cur_log_perp)
                print(
                    ('epoch: %d, train perp: %d, validation classified %d/100 '
                    + '(%.2f epochs/sec)') %
                    (epoch + 1, perp, 100 * (1 - valid_perp_mean), throughput))

                S.save_hdf5('pipc_lstm_%d.pkl' % self.n_hidden, self.model)
                cur_log_perp = self.mod.zeros(())
                prev = now

            sys.stdout.flush()

if __name__ == '__main__':
    trainer = PIPCTrainer(25, offset_timing=10)
    trainer.train()
