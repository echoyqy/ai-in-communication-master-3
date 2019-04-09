import numpy as np

from sklearn.preprocessing import OneHotEncoder

def encode2d(x):

    r = []

    for i in x:
        temp = []
        
        for j in range(len(i)):
            if j % 2 == 0:
                if i[j] == 0.0:
                    if i[j + 1] == 0.0:
                        # r.append(-np.sqrt(2)/2 - np.sqrt(2)/2j)
                        temp.append([-np.sqrt(2)/2,-np.sqrt(2)/2])
                        # temp.append(0)
                    elif i[j + 1] == 1.0:
                        # r.append(-np.sqrt(2)/2 + np.sqrt(2)/2j)
                        temp.append([-np.sqrt(2)/2,np.sqrt(2)/2])
                        # temp.append(1)
                elif i[j] == 1.0:
                    if i[j + 1] == 0.0:
                        # r.append(np.sqrt(2)/2 - np.sqrt(2)/2j)
                        temp.append([np.sqrt(2)/2,-np.sqrt(2)/2])
                        # temp.append(2)
                    elif i[j + 1] == 1.0:
                        # r.append(np.sqrt(2)/2 + np.sqrt(2)/2j)
                        temp.append([np.sqrt(2)/2,np.sqrt(2)/2])
                        # temp.append(3)
        r.append(temp)

    # r = OneHotEncoder().fit_transform(r).toarray()
    # return np.array(r).reshape(shape1,shape2)
    return np.array(r)


def encode2d_onehot(x):
    r = []

    for i in x:
        temp = []

        for j in range(len(i)):
            if j % 2 == 0:
                if i[j] == 0.0:
                    if i[j + 1] == 0.0:
                        # temp.append(-np.sqrt(2)/2 - np.sqrt(2)/2j)
                        temp.append(0)
                    elif i[j + 1] == 1.0:
                        # temp.append(-np.sqrt(2)/2 + np.sqrt(2)/2j)
                        temp.append(1)
                elif i[j] == 1.0:
                    if i[j + 1] == 0.0:
                        # temp.append(np.sqrt(2)/2 - np.sqrt(2)/2j)
                        temp.append(2)
                    elif i[j + 1] == 1.0:
                        # temp.append(np.sqrt(2)/2 + np.sqrt(2)/2j)
                        temp.append(3)
        r.append(temp)

    r = OneHotEncoder().fit_transform(r).toarray()

    return np.array(r)

def encode1d(x):
    r = []
    for i in range(len(x)):
        temp = []
        if i % 2 == 0:
            if x[i] == 0.0:
                if x[i+1] == 0.0:
                    # r.append(-np.sqrt(2)/2 - np.sqrt(2)/2j)
                    r.append(0)
                elif x[i+1] == 1.0:
                    # r.append(-np.sqrt(2)/2 + np.sqrt(2)/2j)
                    r.append(1)
            elif x[i] == 1.0:
                if x[i+1] == 0.0:
                    # r.append(np.sqrt(2)/2 - np.sqrt(2)/2j)
                    r.append(2)
                elif x[i+1] == 1.0:
                    # r.append(np.sqrt(2)/2 + np.sqrt(2)/2j)
                    r.append(3)
    r = np.array(r).reshape(-1,1)
    r = OneHotEncoder().fit_transform(r).toarray()
    return r

# x = np.random.randint(0,2,[1000]).astype(np.float32)
# encode1d(x)