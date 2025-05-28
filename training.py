import ctypes
import data_treatment as dt
from sklearn.metrics import accuracy_score
import numpy as np

#Using ctypes to load the shared library
lib = ctypes.CDLL("./perceptron.so")
#Neuron Function
lib.neuron.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.neuron.restype = ctypes.c_double

#Fit Function
lib.fit.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double, ctypes.c_int]
lib.fit.restype = ctypes.c_double

#Evaluate Accuracy Function
lib.evaluate_accuracy.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
lib.evaluate_accuracy.restype = ctypes.c_double

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
        self.epochs = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.cumulative_error = 0.0
        self.samples = 0 # Will store number of training samples
        self.features = 0 # Will store number of features (excluding bias)
        self.load_data()

#################################################################    

    def learning(self):
        """
        This function trains the perceptron model using the provided data.
        Args:
            epochs (int): Number of epochs for training.
        Returns:
            weights (numpy.ndarray): The learned weights of the perceptron.
            error (float): The final error after training.
        """        
        dimensionality_with_bias = self.features + 1 
        self.weights = np.zeros(dimensionality_with_bias, dtype=np.double)

        bias_train = np.ones((self.samples, 1), dtype=np.double)
        X_train_bias = np.hstack((bias_train, self.X_train))

        acr = 0.0
        self.cumulative_error = 0.0

        while self.accuracy > acr:
            self.epochs += 1
            current_epoch_error_sum = 0.0 
            for xi, target in zip(X_train_bias, self.y_train):
                error = lib.fit(
                    xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                    target, 
                    self.lr,
                    dimensionality_with_bias 
                )
                self.cumulative_error += error
                current_epoch_error_sum += error

            if self.epochs % 10 == 0: 
                acr = self.evaluate()
                print(f"Epoch: {self.epochs}, Cumulative Error during training: {self.cumulative_error}, Accuracy on Test: {acr}\n")
            if self.epochs >= 10000: #Max epochs to prevent infinite loop
                print("Stopping training after 10000 epochs to prevent infinite loop.")
                break

        print(f"Training finished after {self.epochs} epochs.")
        return self.weights, self.cumulative_error
        
    
#################################################################    
    
    def evaluate(self):
        """
        Evaluates the perceptron model on the test data.
        Returns:
            accuracy (float): The accuracy of the model on the test set.
        """
        if self.X_test is None or self.y_test is None or self.weights is None:
            print("Error: Test data or weights not loaded/initialized.")
            return 0.0

        num_test_samples = self.X_test.shape[0]
        
        bias_test = np.ones((num_test_samples, 1), dtype=np.double)
        X_test_bias = np.hstack((bias_test, self.X_test))

        # self.weights and self.y_test should already be np.double from initialization/load_data
        
        acr = lib.evaluate_accuracy(
            X_test_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # X_test with bias
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), # Weights
            self.y_test.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),   # y_test
            num_test_samples,                                              # Correct number of test samples
            self.features + 1           # Number of features including bias (dimensionality of weights and X_test_bias columns)
        )

        return acr

#################################################################

    def load_data(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Loads the training and testing data.
        """
        if X_train is None or y_train is None:
            self.X_train, self.y_train = dt.import_data(False)
        else:
            self.X_train = X_train.astype(np.double) 
            self.y_train = y_train.astype(np.double) 

        if X_test is None or y_test is None:
            self.X_test, self.y_test = dt.import_data(True)
        else:
            self.X_test = X_test.astype(np.double) 
            self.y_test = y_test.astype(np.double) 

        if self.X_train is not None:
            self.samples, self.features = self.X_train.shape 
                                                        
        else:
            print("Warning: self.X_train is None, cannot determine shape.")
            self.samples = 0
            self.features = 0

#################################################################

    def get_epochs(self):
        """
        Returns the number of epochs the model has been trained for.
        Returns:
            epochs (int): The number of epochs.
        """
        return self.epochs
    
#################################################################

    def get_weights(self):
        """
        Returns the learned weights of the perceptron.
        Returns:
            weights (numpy.ndarray): The learned weights.
        """
        return self.weights