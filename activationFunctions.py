import numpy as np

def ELU(x):
    y = np.zeros(x.shape)
    a = 2
    for i in range(x.shape[0]):
        if x[i, 0] > 0.0:
            y[i, 0] = x[i, 0]
        else:
            y[i, 0] = a*(np.e**(x[i, 0])-1)
    return y

def ELUDot(x):
    dx = np.zeros(x.shape)
    a = 2
    for i in range(x.shape[0]):
        if x[i, 0] > 0.0:
            dx[i, 0] = 1
        else:
            dx[i, 0] = a*np.e**(x[i, 0])
    return dx

def RELU(x):
    return 1E-3 * np.maximum(0.0, x)

def RELUDot(x):
    dx = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if x[i, 0] > 0.0:
            dx[i, 0] = 1
        else:
            dx[i, 0] = 5E-2
    return dx


def sigmoid(x):
    return 1 / (1 + np.e**-x)


def sigmoidDot(x):
    return sigmoid(x) * (1 - sigmoid(x))

def af_weight_adjustment(af, x):
    if af == sigmoid:
        return x
    else:
        return x * np.sqrt(1/x.shape[1])