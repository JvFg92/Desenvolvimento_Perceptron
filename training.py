import ctypes
import data_treatment as dt
import numpy as np

#Using ctypes to load the shared library
lib = ctypes.CDLL("./perceptron.so")
lib.perceptron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
lib.perceptron.restype = ctypes.c_double

lib.train_perceptron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double]
lib.train_perceptron.restype = ctypes.c_double

# Load the data:
X_train, y_train, X_test, y_test = dt.load_data()
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

dt.plot_data(X_train, y_train, X_test, y_test)
