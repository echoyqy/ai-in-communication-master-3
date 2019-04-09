import numpy as np


def generate_bit(shape):
    row, col = shape[0], shape[1]

    row_list = []

    for i in range(int(row)):
        # bar.set_description('Processing %d row'%i)
        col_list = list(np.random.randint(0, 2, [int(col), ]))
        row_list.append(col_list)

    return np.array(row_list).astype(np.float32)
