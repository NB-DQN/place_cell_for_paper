from place_cell import PlaceCell


class DeterministicPlaceCell(PlaceCell):
    def move(self, action):
        self.virtual_coordinate = self.neighbor(action)
