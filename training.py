import ctypes
import data_treatment as dt
from sklearn.metrics import accuracy_score
import numpy as np

#Using ctypes to load the shared library
lib = ctypes.CDLL("./perceptron.so")
lib.neuron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.neuron.restype = ctypes.c_double

lib.train_neuron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double]
lib.train_neuron.restype = ctypes.c_double

#################################################################

#This function trains the perceptron model.
def learning(test=False, epochs=30, lr=0.1):
   
    X, y = dt.load_data(test)

    n_samples, n_features = X.shape
    X_bias = np.hstack((np.ones((n_samples, 1)), X))
    weights = np.zeros(X_bias.shape[1], dtype=np.double)

    for epoch in range(epochs):
        for xi, target in zip(X_bias, y):
            error=lib.train_neuron(
                xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),#X
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),#W
                target, 
                lr
            )

    return weights, error

#################################################################

class Perceptron:
    def __init__(self,lr=0.1):
        self.lr = lr
        self.weights = None
    
    def synapse(self, test=False, epochs=30):
        """
        This function trains the perceptron model using the provided data.
        Args:
            test (bool): If True, loads the test data; otherwise, loads the training data.
            epochs (int): Number of epochs for training.
        Returns:
            weights (numpy.ndarray): The learned weights of the perceptron.
            error (float): The final error after training.
        """
        X, y = dt.load_data(test)

        n_samples, n_features = X.shape
        X_bias = np.hstack((np.ones((n_samples, 1)), X))
        self.weights = np.zeros(X_bias.shape[1], dtype=np.double)

        for epoch in range(epochs):
            for xi, target in zip(X_bias, y):
                error=lib.train_neuron(
                    xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),#X
                    self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),#W
                    target, 
                    self.lr
                )

        return self.weights, error