import ctypes
import data_treatment as dt
import numpy as np

#Using ctypes to load the shared library
lib = ctypes.CDLL("./perceptron.so")
lib.perceptron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
lib.perceptron.restype = ctypes.c_double

lib.train_perceptron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double]
lib.train_perceptron.restype = ctypes.c_double

#This function trains the perceptron model.
def learning(X, y, epochs=30, lr=0.1):

    n_samples, n_features = X.shape
    X_bias = np.hstack((np.ones((n_samples, 1)), X))
    weights = np.zeros(X_bias.shape[1], dtype=np.double)

    for epoch in range(epochs):
        for xi, target in zip(X_bias, y):
            error=lib.train_perceptron(
                xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(xi),
                target, 
                lr
            )

    return weights, error
