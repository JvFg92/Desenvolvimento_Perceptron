import ctypes
import data_treatment as dt
from sklearn.metrics import accuracy_score
import numpy as np

#Using ctypes to load the shared library
lib = ctypes.CDLL("./perceptron.so")
lib.neuron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.neuron.restype = ctypes.c_double

lib.fit.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double]
lib.fit.restype = ctypes.c_double

#################################################################

class Perceptron:
    def __init__(self,lr=0.1, accuracy=0.9):
        """
        Initializes the Perceptron model with a learning rate and accuracy threshold.
        Args:
            lr (float): Learning rate for weight updates.
            accuracy (float): Desired accuracy threshold for training.
        """
        self.accuracy = accuracy
        self.lr = lr
        self.weights = None
    
    def learning(self, epochs=30):
        """
        This function trains the perceptron model using the provided data.
        Args:
            epochs (int): Number of epochs for training.
        Returns:
            weights (numpy.ndarray): The learned weights of the perceptron.
            error (float): The final error after training.
        """
        X, y = dt.load_data(False)

        n_samples, n_features = X.shape
        X_bias = np.hstack((np.ones((n_samples, 1)), X))
        self.weights = np.zeros(X_bias.shape[1], dtype=np.double)

        for epoch in range(epochs):
            for xi, target in zip(X_bias, y):
                error=lib.fit(
                    xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),#X
                    self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),#W
                    target, 
                    self.lr
                )

        return self.weights, error
    
    