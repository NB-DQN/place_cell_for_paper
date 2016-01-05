from place_cell import PlaceCell
from os import path
import chainer
import chainer.functions as F
import pickle
import numpy as np


class VisualPredictivePlaceCell(PlaceCell):
    def __init__(self, size):
        super(VisualPredictivePlaceCell, self).__init__(size)

        self.n_units = 60
        self.state = self.make_initial_state(batchsize=1, train=False)

        model_dir = path.join(path.dirname(__file__), 'pickles')
        lstm_model = path.join(model_dir, 'vppc_lstm_%d.pkl' % self.n_units)
        with open(lstm_model, 'rb') as file:
            self.lstm = pickle.load(file)
        svm_model = path.join(model_dir, 'vppc_svm_%d.pkl' % self.n_units)
        with open(svm_model, 'rb') as file:
            self.svm = pickle.load(file)

        # initialize with visual image obtained on (0, 0)
        self.predicted_visual_image = np.array([
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    def make_initial_state(self, batchsize=1, train=True):
        return {
            name: chainer.Variable(np.zeros((batchsize, self.n_units),
                                   dtype=np.float32), volatile=not train)
            for name in ('c', 'h')}

    def move(self, action, visual_image=None):
        action_units = [0, 0, 0, 0]
        action_units[action] = 1

        if visual_image is None:
            data = np.array(
                [action_units + self.predicted_visual_image.tolist()],
                dtype='float32')
        else:
            data = np.array(
                [action_units + visual_image.tolist()], dtype='float32')
        x = chainer.Variable(data, volatile=True)
        h_in = self.lstm.x_to_h(x) + self.lstm.h_to_h(self.state['h'])
        c, h = F.lstm(self.state['c'], h_in)
        self.state = {'c': c, 'h': h}

        y = self.lstm.h_to_y(h)
        sigmoid_y = 1 / (1 + np.exp(-y.data))
        self.predicted_visual_image = \
            np.round((np.sign(sigmoid_y - 0.5) + 1) / 2)[0]

        coordinate_id = self.svm.predict(h.data[0])[0]
        self.set_coordinate_id(coordinate_id)

        return self.virtual_coordinate

    def h_to_coordinates(self, h):
        return self.svm.predict(h)[0]
