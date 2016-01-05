from place_cell import PlaceCell
import numpy as np


class DeterministicPlaceCell(PlaceCell):
    def move(self, action):
        self.virtual_coordinate = self.neighbor(action)
        output = np.zeros(
            self.environment_size[0] * self.environment_size[1],
            dtype=np.bool)
        output[self.coordinate_id()] = 1
