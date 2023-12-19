from typing import NewType

import numpy as np


Input = NewType('Input', int)


def act(x):
    return 0 if x < 0.5 else 1


def go(house: Input, rock: Input, attr: Input):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print('Значение сумм на нейронах скрытого слоя {0}'.format(sum_hidden))
    out_hidden = np.array([act(x) for x in sum_hidden])
    print('Значение сумм на выходах нейронов скрытого слоя {0}'.format(out_hidden))
    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print('Выходное значение: {0}'.format(y))
    return y


if __name__ == '__main__':
    go(
        Input(1),
        Input(0),
        Input(1),
    )
