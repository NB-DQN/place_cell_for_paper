import numpy as np
import math
import random


class DatasetGenerator:
    def __init__(self, environment_size):
        self.environment_size = environment_size
        self.current_coordinate = (0, 0)
        self.DEGREE_PER_DOT = 6

    def neighbor(self, action):
        neighbors = [
            (self.current_coordinate[0] + 1, self.current_coordinate[1]    ),
            (self.current_coordinate[0] - 1, self.current_coordinate[1]    ),
            (self.current_coordinate[0]    , self.current_coordinate[1] + 1),
            (self.current_coordinate[0]    , self.current_coordinate[1] - 1)]
        return neighbors[action]

    def validate_action(self, action):
        coordinate = self.neighbor(action)
        return 0 <= coordinate[0] < self.environment_size[0] and \
               0 <= coordinate[1] < self.environment_size[1]

    def coordinate_id(self, coordinate=None):
        if coordinate is None:
            coordinate = self.current_coordinate
        return coordinate[0] + coordinate[1] * self.environment_size[0]

    def get_coordinate_from_id(self, coordinate_id):
        x = coordinate_id % self.environment_size[0]
        y = (coordinate_id - x) / self.environment_size[0]
        return (x, y)

    def visual_image(self, coordinate_id=None):
        if coordinate_id is None:
            coordinate_id = self.coordinate_id()
        coordinate = self.get_coordinate_from_id(coordinate_id)

        image = np.zeros(360 / self.DEGREE_PER_DOT, dtype='int32')

        visual_targets = [
            (self.environment_size[0], self.environment_size[1]),
            (                      -1, self.environment_size[1]),
            (                      -1,                       -1),
            (self.environment_size[0],                       -1)]
        for target in visual_targets:
            distance = math.sqrt(
                (coordinate[0] - target[0]) ** 2 +
                (coordinate[1] - target[1]) ** 2)
            visual_width = math.degrees(math.atan(0.5 / distance))
            angle = math.degrees(math.atan2(
                target[1] - coordinate[1], target[0] - coordinate[0]))
            if angle < 0:
                angle += 360

            visual_range = \
                [round(i / self.DEGREE_PER_DOT)
                 for i in [angle - visual_width, angle + visual_width]]
            image[visual_range[0]:(visual_range[1] + 1)] = 1
        return image.tolist()

    def generate_sequence(self, sequence_length,
                          offset_timing=1, go_away_from_start=False):
        image_units = []
        action_units = []
        coordinates = []

        image_units.append(self.visual_image())
        coordinates.append(self.current_coordinate)

        for i in range(0, sequence_length):
            action_candidates = list(range(4))

            if go_away_from_start:
                threshold = 0.2
                if self.current_coordinate[0] == 4 and \
                   self.current_coordinate[1] <= 4 and \
                   random.random() > threshold:
                    action_candidates.remove(1)
                if self.current_coordinate[1] == 4 and \
                   self.current_coordinate[0] <= 4 and \
                   random.random() > threshold:
                    action_candidates.remove(3)

            while True:
                action = random.choice(action_candidates)
                if self.validate_action(action):
                    break
            self.current_coordinate = self.neighbor(action)

            units = [0, 0, 0, 0]
            units[action] = 1
            action_units.append(units)
            image_units.append(self.visual_image())
            coordinates.append(self.current_coordinate)

        image_units_offset = []
        coordinate_ids = []
        for i, coordinate in enumerate(coordinates):
            coordinate_ids.append(self.coordinate_id(coordinate))
            if (coordinate[0] + coordinate[1]) % offset_timing == 0:
                image_units_offset.append(image_units[i])
            else:
                image_units_offset.append([0] * int(360 / self.DEGREE_PER_DOT))
        return {
            'sequence_length':    sequence_length,
            'offset_timing':      offset_timing,
            'go_away_from_start': go_away_from_start,
            'image_units':        image_units,
            'image_units_offset': image_units_offset,
            'action_units':       action_units,
            'coordinate_ids':     coordinate_ids}
