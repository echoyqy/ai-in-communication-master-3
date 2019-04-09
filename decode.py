import numpy as np


def decode2d(y):
    x = []
    for i in y:
        temp = []
        for j in range(len(i)):
            if i[j][0] > 0 and i[j][1] > 0 :

                temp.append(1.0)
                temp.append(1.0)

            elif i[j][0] > 0 and i[j][1] < 0 :

                temp.append(1.0)
                temp.append(0.0)
            
            elif i[j][0] < 0 and i[j][1] < 0 :
                temp.append(0.0)
                temp.append(0.0)
            
            elif i[j][0] < 0 and i[j][1] > 0 :
                temp.append(0.0)
                temp.append(1.0)
        x.append(temp)
    return np.array(x).astype(np.float32)
