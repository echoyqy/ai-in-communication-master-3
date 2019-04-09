import numpy as np

temp = []


def decode(x):
    x = x.astype(np.float32)
    for i in range(len(x)):
        if x[i]==0.0:
            temp.append([0.0, 0.0])
        elif x[i]==1.0:
            temp.append([0.0, 1.0])
        elif x[i]==2.0:
            temp.append([1.0, 0.0])
        elif x[i]==3.0:
            temp.append([1.0, 1.0])
    return np.array(temp).astype(np.float32)

