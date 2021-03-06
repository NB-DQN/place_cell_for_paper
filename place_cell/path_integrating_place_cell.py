from place_cell import PlaceCell
from os import path
import chainer
import chainer.functions as F
import pickle
import numpy as np


class PathIntegratingPlaceCell(PlaceCell):
    def __init__(self, size):
        super(PathIntegratingPlaceCell, self).__init__(size)

        self.offset_timing = 2
        self.state = self.make_initial_state(batchsize=1, train=False)

        model_dir = path.join(path.dirname(__file__), 'pickles')
        lstm_model = path.join(model_dir, 'pipc_%d.pkl' % self.offset_timing)
        with open(lstm_model, 'rb') as file:
            self.lstm = pickle.load(file)

    def make_initial_state(self, batchsize=1, train=True):
        return {
            name: chainer.Variable(np.zeros((batchsize, 25), dtype=np.float32),
                                   volatile=not train)
            for name in ('c', 'h')}

    def move(self, action, precise_coordinate=None):
        action_units = [0, 0, 0, 0]
        action_units[action] = 1

        coordinate_units = [0] * 81
        coordinate = precise_coordinate or self.virtual_coordinate
        if (coordinate[0] + coordinate[1]) % self.offset_timing == 0:
            coordinate_units[self.coordinate_id(coordinate)] = 1

        data = np.array([action_units + coordinate_units], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.lstm.x_to_h(x) + self.lstm.h_to_h(self.state['h'])
        c, h = F.lstm(self.state['c'], h_in)
        self.state = {'c': c, 'h': h}

        y = self.lstm.h_to_y(h)
        exp_y = np.exp(y.data[0])
        softmax_y = exp_y / exp_y.sum(axis=0, keepdims=True)
        coordinate_id = softmax_y.argmax()
        self.set_coordinate_id(coordinate_id)

        return self.virtual_coordinate
