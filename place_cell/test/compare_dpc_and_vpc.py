from visual_place_cell import VisualPlaceCell
from deterministic_place_cell import DeterministicPlaceCell

import random
from matplotlib import pyplot as plt

size = (9, 9)

res = []
for j in range(1000):
    vpc = VisualPlaceCell(size)
    dpc = DeterministicPlaceCell(size)
    for i in range(100):
        while True:
            a = random.choice(range(4))
            if dpc.validate_action(a):
                break

        vpc.move(a)
        dpc.move(a)

        if vpc.virtual_coordinate != dpc.virtual_coordinate:
            res.append(i)
            break
    else:
        res.append(100)
        pass

    print(j)

plt.hist(res, bins=100)
plt.show()
print(min(res))
