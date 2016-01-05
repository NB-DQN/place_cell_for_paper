from abc import ABCMeta, abstractmethod


class PlaceCell(object):
    __metaclass__ = ABCMeta

    def __init__(self, environment_size):
        self.environment_size = environment_size
        self.virtual_coordinate = (0, 0)

    def neighbor(self, action):
        neighbors = [
            (self.virtual_coordinate[0] + 1, self.virtual_coordinate[1]    ),
            (self.virtual_coordinate[0] - 1, self.virtual_coordinate[1]    ),
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] + 1),
            (self.virtual_coordinate[0]    , self.virtual_coordinate[1] - 1)]
        return neighbors[action]

    def validate_action(self, action):
        coordinate = self.neighbor(action)
        return 0 <= coordinate[0] < self.environment_size[0] and \
               0 <= coordinate[1] < self.environment_size[1]

    def coordinate_id(self):
        return self.virtual_coordinate[0] + \
               self.virtual_coordinate[1] * self.environment_size[0]

    def set_coordinate_id(self, coordinate_id):
        new_x = coordinate_id % self.environment_size[0]
        new_y = (coordinate_id - new_x) / self.environment_size[0]
        self.virtual_coordinate = (new_x, new_y)

    @abstractmethod
    def move(self, action):
        pass
