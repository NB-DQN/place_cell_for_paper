"""
Base class for pre-training place cells

Model:
    LSTM with one hidden layer
    Necessities of truncated BPTT or gradient clip are unclear.

NOTE: only tested under python3 environment
"""

import numpy as np

from chainer import Chain
from chainer import Variable
from chainer import links as L
from chainer import optimizers as O
from chainer import optimizer
from chainer import cuda

from abc import ABCMeta, abstractmethod

from dataset_generator import DatasetGenerator


class LSTMTrainer:
    __metaclass__ = ABCMeta

    class LSTM(Chain):
        def __init__(self, n_input, n_hidden, n_output, train=True):
            super(LSTMTrainer.LSTM, self).__init__(
                link1=L.LSTM(n_input, n_hidden),
                link2=L.Linear(n_hidden, n_output))
            self.train = train

        def reset_state(self):
            self.link1.reset_state()

        def __call__(self, x):
            h = self.link1(x)
            y = self.link2(h)
            return h, y

    def __init__(self, n_hidden, n_epoch,
                 environment_size=(9, 9), sequence_length=100, offset_timing=1,
                 gpu=-1):
        self.n_hidden = n_hidden  # number of units in hidden layer
        self.n_epoch = n_epoch    # number of epochs
        self.batchsize = 1        # minibatch size
        self.backprop_length = 1  # length of truncated BPTT
        self.gradient_clip = 5    # gradient norm threshold to clip

        self.environment_size = environment_size
        self.sequence_length = sequence_length
        self.offset_timing = offset_timing
        self.gpu = gpu

        self.train_errors = []
        self.valid_errors_mean = []
        self.valid_errors_se = []

    def setup_model(self): #, classifier):
        self.model = LSTMTrainer.LSTM(64, self.n_hidden, self.n_output)
        for param in self.model.params():
            data = param.data
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)
        self.optimizer = O.SGD(lr=1.)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(optimizer.GradientClipping(self.gradient_clip))

    def setup_gpu(self):
        if self.gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(self.gpu).use()
            self.model.to_gpu()
            self.mod = cuda.cupy
        else:
            self.mod = np

    def setup_dataset(self):
        self.dataset_generator = DatasetGenerator(self.environment_size)
        self.sequence_length = self.sequence_length
        self.offset_timing = self.offset_timing
        self.validation_timing = self.n_epoch / 40
        self.validation_dataset_length = 20
        self.validation_dataset = [
            self.generate_data()
            for i in range(self.validation_dataset_length)]
        self.test_data = self.generate_data()

    def validate(self, epoch, train_data, cur_log_perp):
        train_perp = self.evaluate(train_data)
        self.train_errors.append(train_perp)

        valid_perps = self.mod.zeros(self.validation_dataset_length)
        for i in range(self.validation_dataset_length):
            valid_perps[i] = self.evaluate(self.validation_dataset[i])

        valid_perp_mean = np.mean(valid_perps, axis=0)
        self.valid_errors_mean.append(valid_perp_mean)

        valid_perp_se = \
            np.std(valid_perps, axis=0) / np.sqrt(self.validation_timing)
        self.valid_errors_se.append(valid_perp_se)

        if epoch == 0:
            perp = 0
        else:
            perp = cuda.to_cpu(cur_log_perp) / self.validation_timing
            perp = int(perp * 100) / 100.0

        if epoch >= self.n_epoch / 4:
            self.optimizer.lr /= 1.2
            print('learning rate: %.3f' % self.optimizer.lr)

        return train_perp, valid_perp_mean, valid_perp_se, perp

    def toVariable(self, x, dtype='int32', train=True):
        return Variable(self.mod.asarray([x], dtype=dtype), volatile=not train)

    @abstractmethod
    def generate_data(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def train(self):
        pass
