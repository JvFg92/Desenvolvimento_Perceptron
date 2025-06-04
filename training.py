import ctypes
import data_treatment as dt
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

#Recall Score Function
lib.recall.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
lib.recall.restype = ctypes.c_double

#Predict Function
lib.predict.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
lib.predict.restype = ctypes.POINTER(ctypes.c_int)



#################################################################

class Perceptron:
    def __init__(self,lr=0.1, accuracy=0.9, generated = False, linear=False):
        """
        Initializes the Perceptron model with a learning rate and accuracy threshold.
        Args:
            lr (float): Learning rate for weight updates.
            accuracy (float): Desired accuracy threshold for training.
        """
        self.generated = generated
        self.linear = linear
        self.ref_accuracy = accuracy
        self.lr = lr
        self.weights = None
        self.epochs = 0
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.cumulative_error = 0.0
        self.samples = 0
        self.features = 0 #number of features (excluding bias)
        self.test_accuracy = 0.0
        self.recall_score = 0.0
        self.recall_history = []
        self.train_accuracies = [] 
        self.train_epochs = []
        self.weights_history = []
        self.train_errors = []
        self.load_data(gnt=generated, lnr=linear)

#################################################################

    def __del__(self):
        """
        Destructor to clean up resources.
        """
        print("Perceptron resources cleaned up.")

#################################################################

    def learning(self, epcs=1000):
        """
        This function trains the perceptron model using the provided data.
        Args:
            epochs (int): Number of epochs for training.
        Returns:
            weights (numpy.ndarray): The learned weights of the perceptron.
            error (float): The final error after training.
        """        
        print("Starting training...\n")
        dimensionality_with_bias = self.features + 1 
        self.weights = np.zeros(dimensionality_with_bias, dtype=np.double)

        bias_train = np.ones((self.samples, 1), dtype=np.double)
        X_train_bias = np.hstack((bias_train, self.X_train))

        self.test_accuracy = 0.0
        self.cumulative_error = 0.0
        self.recall_score = 0.0
        self.recall_history = []
        self.train_epochs = []
        self.train_accuracies = []
        self.weights_history = []
        self.train_errors = []

        while self.ref_accuracy > self.test_accuracy and self.epochs < epcs:
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

            if self.epochs % 10 == 0 : self.test_accuracy, self.recall_score = self.evaluate(self.X_train, self.y_train)
            #self.test_accuracy = self.evaluate(self.X_train, self.y_train) #May converge too fast in some cases.
            acr,rcl= self.evaluate(self.X_train, self.y_train)
            self.train_accuracies.append(acr)
            self.recall_history.append(rcl)
            self.train_epochs.append(self.epochs)
            self.weights_history.append(self.weights.copy())
            self.train_errors.append(error)
            self.cumulative_error = current_epoch_error_sum / self.samples
            print(f"Epoch {self.epochs}: Cumulative Error Normalized = {self.cumulative_error:.4f}, Training Accuracy = {self.train_accuracies[-1]}, Recall Score = {self.recall_history[-1]:.4f}")

        self.test_accuracy, self.recall_score = self.evaluate(self.X_test, self.y_test)
        print(f"\nFinal Training Accuracy (Tested) after {self.epochs} epochs: {self.test_accuracy}, Recall Score: {self.recall_score:.4f}")
        
#################################################################

    def evaluate(self,X,y):
        """
        Evaluates the perceptron model on the test data.
        Returns:
            accuracy (float): The accuracy of the model on the test set.
            recall (float): The recall score of the model on the test set.
        """
        if X is None or y is None or self.weights is None:
            print("Error: Test data or weights not loaded/initialized.")
            return 0.0

        num_test_samples = X.shape[0]
        
        bias_test = np.ones((num_test_samples, 1), dtype=np.double)
        X_test_bias = np.hstack((bias_test, X))

        acr = lib.evaluate_accuracy(
            X_test_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # X_test with bias
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), # Weights
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),   # y_test
            num_test_samples,                                              
            self.features + 1           # Number of features including bias
        )
        
        recall= lib.recall(
            X_test_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),              
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),                                               
            num_test_samples,                                               
            self.features + 1                                       
        )

        return acr, recall

#################################################################

    def think(self, X=None):
        """
        Predicts the class labels for the input data using the trained model.
        Args:
            X (numpy.ndarray): Input data for prediction. If None, uses self.X_test.
        Returns:
            numpy.ndarray: Predicted class labels.
        """
        if X is None:
            if self.X_test is None:
                print("Error: No test data available for prediction.")
                return None
            X = self.X_test

        num_samples = X.shape[0]
        bias = np.ones((num_samples, 1), dtype=np.double)
        X_bias = np.hstack((bias, X))

        predictions = lib.predict(
            X_bias.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
            num_samples, 
            self.features + 1
        )
        results = np.ctypeslib.as_array(predictions, shape=(num_samples,))
        return results

#################################################################

    def load_data(self, X_train=None, y_train=None, X_test=None, y_test=None , gnt=False, lnr=False):
        """
        Loads the training and testing data.
        """
        if X_train is None or y_train is None:
            if gnt: self.X_train, self.y_train = dt.generate_diagnostics(False, lnr)
            else: self.X_train, self.y_train = dt.import_data(False,lnr)
        else:
            self.X_train = X_train.astype(np.double) 
            self.y_train = y_train.astype(np.double) 

        if X_test is None or y_test is None:
            if gnt: self.X_test, self.y_test = dt.generate_diagnostics(True, lnr)
            else: self.X_test, self.y_test = dt.import_data(True,lnr)
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

    def cross_validate(self, k=5, plot=False):
        """
        Performs k-fold cross-validation on the training data.
        Args:
            k (int): Number of folds for cross-validation.
        Returns:
            list: List of accuracies for each fold.
            list: List of recall scores for each fold.
        """
        if self.X_train is None or self.y_train is None:
            print("Error: Original training data not loaded into the Perceptron instance.")
            return []

        #Preserve original Instance
        original_X_train = self.X_train.copy()
        original_y_train = self.y_train.copy()
        original_X_test = self.X_test.copy() if self.X_test is not None else None
        original_y_test = self.y_test.copy() if self.y_test is not None else None

        original_weights = self.weights.copy() if self.weights is not None else None
        original_epochs_count = self.epochs 
        original_cumulative_error = self.cumulative_error
        original_test_accuracy = self.test_accuracy
        original_recall_score = self.recall_score
        
        original_train_accuracies_history = list(self.train_accuracies)
        original_recall_history = list(self.recall_history)
        original_train_epochs_history = list(self.train_epochs)
        original_weights_history_log = list(self.weights_history)
        original_train_errors_history = list(self.train_errors)
        
        original_samples = self.samples
        original_features = self.features

        X_internal_cv = original_X_train 
        y_internal_cv = original_y_train
        
        fold_accuracies = []
        fold_recall_history = []
        fold_recall_error = []
        num_total_samples_cv = X_internal_cv.shape[0]
        fold_size = num_total_samples_cv // k
        
        print(f"\nStarting {k}-Fold Cross-Validation...")

        for i in range(k):
            print(f"\n--- Cross-Validation Fold {i+1}/{k} ---")
            start = i * fold_size
            end = (start + fold_size) if i < k - 1 else num_total_samples_cv

            X_val_fold = X_internal_cv[start:end]
            y_val_fold = y_internal_cv[start:end]

            X_train_fold = np.concatenate((X_internal_cv[:start], X_internal_cv[end:]), axis=0)
            y_train_fold = np.concatenate((y_internal_cv[:start], y_internal_cv[end:]), axis=0)

            #Load data for the current fold:
            self.load_data(X_train=X_train_fold, y_train=y_train_fold, X_test=X_val_fold, y_test=y_val_fold)
            
            #Reset parameters and history for a clean training run for this fold
            self.epochs = 0 
            self.cumulative_error = 0.0
            self.test_accuracy = 0.0 
            self.recall_score = 0.0
            self.recall_history = []
            self.train_accuracies = []
            self.train_epochs = []
            self.weights_history = []
            self.train_errors = []

            self.learning() 
             
            fold_accuracies.append(self.test_accuracy)
            fold_recall_history.append(self.recall_score)
            fold_recall_error.append(self.cumulative_error)

        #Restore original state of the Perceptron instance
        self.X_train = original_X_train
        self.y_train = original_y_train
        self.X_test = original_X_test
        self.y_test = original_y_test
        self.samples = original_samples
        self.features = original_features
        
        self.weights = original_weights
        self.epochs = original_epochs_count
        self.cumulative_error = original_cumulative_error
        self.test_accuracy = original_test_accuracy 
        
        self.train_accuracies = original_train_accuracies_history
        self.recall_history = original_recall_history
        self.recall_score = original_recall_score
        self.train_epochs = original_train_epochs_history
        self.weights_history = original_weights_history_log
        self.train_errors = original_train_errors_history
        
        print("\n--- Cross-Validation Summary ---")
        print(f"Fold accuracies: {[f'{acc*100:.2f}%' for acc in fold_accuracies]}")
        print(f"Mean CV accuracy: {np.mean(fold_accuracies)*100:.2f}%")
        print(f"Fold recall scores: {[f'{rec*100:.2f}' for rec in fold_recall_history]}")
        print(f"Mean CV recall: {np.mean(fold_recall_history)*100:.2f}%")

        if plot:
            self.plot_accuracy(acr=fold_accuracies, epc=range(1, k + 1), ttl=f"{k}-Fold Cross-Validation Accuracy", xlabel="Fold Number")
            self.plot_recall(rcl=fold_recall_history, epc=range(1, k + 1), ttl=f"{k}-Fold Cross-Validation Recall", xlabel="Fold Number")
            self.plot_errors(error=fold_recall_error, epc=range(1, k + 1))
        print("\nPerceptron state restored to pre-cross-validation.")

#################################################################

    def get_post_train(self, plot=False):
        """
        Returns the final training results after training.
        """
        if plot:
            self.plot_errors()
            self.plot_weights()
            self.plot_accuracy(acr=self.train_accuracies, epc=self.train_epochs, ttl="Training Accuracy Over Epochs")
            self.plot_recall(rcl=self.recall_history, epc=self.train_epochs, ttl="Training Recall Over Epochs")
            self.plot_decision_boundary()
        return self.weights, self.cumulative_error, self.test_accuracy, self.epochs, self.recall_score

#################################################################

    def plot_data(self):
        """
        Plots the training and testing data.
        """
        if self.generated and self.linear:
            dt.plot_data(self.X_train, self.y_train, self.X_test, self.y_test, title="Training and Testing Data [Tumor Classification Benign(0)/Malignant(1)]", xlabel="Morning Size", ylabel="Color Intensity")
        elif self.generated and not self.linear:
            dt.plot_data(self.X_train, self.y_train, self.X_test, self.y_test, title="Training and Testing Data [Tumor Classification Benign(0)/Malignant(1)]", xlabel="Morning Size", ylabel="Color Intensity")
        elif not self.generated and self.linear:
            dt.plot_data(self.X_train, self.y_train, self.X_test, self.y_test, title="Training and Testing Data [Iris Classification Setosa(1)/No Setosa(0)]", xlabel="Sepal Length", ylabel="Sepal Width")
        else:
            dt.plot_data(self.X_train, self.y_train, self.X_test, self.y_test, title="Training and Testing Data [Iris Classification Versicolor(1)/Virginica(0)]", xlabel="Sepal Length", ylabel="Sepal Width")

#################################################################
    
    def plot_decision_boundary(self):
        """
        Plots the decision boundary of the trained model.
        """
        if self.generated:
            dt.plot_decision_boundary(self.X_train, self.y_train, self.weights, title="Decision Boundary (Tumor Classification)", xlabel="Morning Size", ylabel="Color Intensity")
        else:
            dt.plot_decision_boundary(self.X_train, self.y_train, self.weights, title="Decision Boundary (Iris Classification)", xlabel="Sepal Length", ylabel="Sepal Width")

#################################################################
    
    def plot_accuracy(self, acr = None, epc=None, ttl="Model Accuracy Over Epochs", ylabel="Accuracy", xlabel="Epochs"):
        """
        Plots the accuracy of the model over epochs.
        """
        if acr is None: acr = self.train_accuracies
        if epc is None: epc = self.train_epochs
        dt.general_plot(data=acr, interval=epc, title=ttl, ylabel=ylabel, xlabel=xlabel)

#################################################################

    def plot_weights(self):
        """
        Plots the evolution of weights during training.
        """
        dt.plot_weights(self.weights_history, self.train_epochs, self.features)

#################################################################

    def plot_errors(self, error=None, epc=None):
        """
        Plots the training errors over epochs.
        """
        if error is None: error = self.train_errors
        if epc is None: epc = self.train_epochs
        dt.general_plot(data=error, interval=epc, title="Training Errors Over Epochs", ylabel="Error", xlabel="Epochs")

##################################################################

    def plot_recall(self, rcl = None, epc=None, ttl="Model Recall Over Epochs", ylabel="Recall", xlabel="Epochs"):
        """
        Plots the recall of the model over epochs.
        """
        if rcl is None: rcl = self.train_recall_history
        if epc is None: epc = self.train_epochs
        dt.general_plot(data=rcl, interval=epc, title=ttl, ylabel=ylabel, xlabel=xlabel)

##################################################################