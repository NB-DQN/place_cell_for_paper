import numpy as np
import math
import random

class DatasetGenerator:
    def __init__(self, size):
        self.size = size
        self.current_coordinate = (0, 0)

    def coordinate_id(self, coordinate=None):
        if coordinate is None:
            coordinate = self.current_coordinate
        return coordinate[0] + coordinate[1] * self.size[0]

    def get_coordinate_from_id(self, cid):
        x = cid % self.size[0]
        y = (cid - x) / self.size[0]
        return (x, y)

    def visual_targets(self):
        return [ \
            (self.size[0], self.size[1]), \
            (          -1, self.size[1]), \
            (          -1,           -1), \
            (self.size[0],           -1)]

    def visual_image(self, cid=None):
        if cid is None:
            cid = self.coordinate_id()
        coordinate = self.get_coordinate_from_id(cid)

        DEGREE_PER_DOT = 6

        image = np.zeros(360 / DEGREE_PER_DOT)
        for target in self.visual_targets():
            distance = math.sqrt( \
                (coordinate[0] - target[0]) ** 2 + \
                (coordinate[1] - target[1]) ** 2)
            visual_width = math.degrees(math.atan(0.5 / distance))
            angle = math.degrees(math.atan2(target[1] - coordinate[1], target[0] - coordinate[0]))
            if angle < 0:
                angle += 360

            visual_range = [round(i / DEGREE_PER_DOT) for i in [angle - visual_width, angle + visual_width]]
            image[visual_range[0]:(visual_range[1] + 1)] = 1
        return image

    def generate_seq(self, seq_length, offset_timing):
        image = []
        directions = []
        coordinates = []

        image.append(self.visual_image())
        coordinates.append(0)

        for i in range(0, seq_length):
            direction_choice = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            if self.current_coordinate[0] == self.size[0] - 1:
                direction_choice.remove([1, 0, 0, 0])
            if self.current_coordinate[0] == 0:
                direction_choice.remove([0, 1, 0, 0])
            if self.current_coordinate[1] == self.size[1] - 1:
                direction_choice.remove([0, 0, 1, 0])
            if self.current_coordinate[1] == 0:
                direction_choice.remove([0, 0, 0, 1])
            direction = random.choice(direction_choice)

            if   direction == [1, 0, 0, 0]:
                self.current_coordinate = (self.current_coordinate[0] + 1, self.current_coordinate[1]    )
            elif direction == [0, 1, 0, 0]:
                self.current_coordinate = (self.current_coordinate[0] - 1, self.current_coordinate[1]    )
            elif direction == [0, 0, 1, 0]:
                self.current_coordinate = (self.current_coordinate[0]    , self.current_coordinate[1] + 1)
            elif direction == [0, 0, 0, 1]:
                self.current_coordinate = (self.current_coordinate[0]    , self.current_coordinate[1] - 1)

            directions.append(direction)
            image.append(self.visual_image())
            coordinates.append(self.coordinate_id())

        input = []
        for i in range(len(directions)):
            if coordinates[i] % offset_timing == 0: 
                input.append(directions[i] + image[i].tolist())
            else:
                input.append(directions[i] + [0] * 60)

        return { 'input': input, 'output': coordinates[1:]}
        
    def generate_seq_remote(self, seq_length, offset_timing):
        image = []
        directions = []
        coordinates = []

        image.append(self.visual_image())
        coordinates.append(0)

        for i in range(0, seq_length):
            direction_choice = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            if self.current_coordinate[0] == self.size[0] - 1:
                direction_choice.remove([1, 0, 0, 0])
            if self.current_coordinate[0] == 0:
                direction_choice.remove([0, 1, 0, 0])
            if self.current_coordinate[1] == self.size[1] - 1:
                direction_choice.remove([0, 0, 1, 0])
            if self.current_coordinate[1] == 0:
                direction_choice.remove([0, 0, 0, 1])
                
            if current[0] == 4 and current[1] <= 4:
                threshold = 0.2
                if random.random() > threshold:
                    direction_choice.remove([0, 1, 0, 0])
            if current[1] == 4 and current[0] <= 4:
                threshold = 0.2
                if random.random() > threshold:
                    direction_choice.remove([0, 0, 0, 1])
                
            direction = random.choice(direction_choice)

            if   direction == [1, 0, 0, 0]:
                self.current_coordinate = (self.current_coordinate[0] + 1, self.current_coordinate[1]    )
            elif direction == [0, 1, 0, 0]:
                self.current_coordinate = (self.current_coordinate[0] - 1, self.current_coordinate[1]    )
            elif direction == [0, 0, 1, 0]:
                self.current_coordinate = (self.current_coordinate[0]    , self.current_coordinate[1] + 1)
            elif direction == [0, 0, 0, 1]:
                self.current_coordinate = (self.current_coordinate[0]    , self.current_coordinate[1] - 1)

            directions.append(direction)
            image.append(self.visual_image())
            coordinates.append(self.coordinate_id())

        input = []
        for i in range(len(directions)):
            if coordinates[i] % offset_timing == 0: 
                input.append(directions[i] + image[i].tolist())
            else:
                input.append(directions[i] + [0] * 60)

        return { 'input': input, 'output': coordinates[1:]}

