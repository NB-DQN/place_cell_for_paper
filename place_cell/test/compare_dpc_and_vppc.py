import os
import sys
import random
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual_predictive_place_cell import VisualPredictivePlaceCell
from deterministic_place_cell import DeterministicPlaceCell


size = (9, 9)

res = []
for j in range(1000):
    vppc = VisualPredictivePlaceCell(size)
    dpc = DeterministicPlaceCell(size)
    for i in range(100):
        while True:
            a = random.choice(range(4))
            if dpc.validate_action(a):
                break

        vppc.move(a)
        dpc.move(a)

        if vppc.virtual_coordinate != dpc.virtual_coordinate:
            res.append(i)
            break
    else:
        res.append(100)
        pass

    if (j + 1) % 100 == 0:
        print j + 1
    else:
        sys.stdout.write('.')
        sys.stdout.flush()

plt.hist(res, bins=100)
plt.show()
