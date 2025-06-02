from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

def import_data(test=False, linear=False):
  """
    This function loads the iris dataset, converts it to a binary classification problem,
    and splits it into training and testing sets.
    Returns:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
  """
  
  iris = load_iris()
  X = iris.data
  y = iris.target

  if linear:
    
    y_processed = np.where(y == 0.0, 1.0, 0.0)  
    #Convert to binary classification problem
    X_processed = X[:, :2] #Using only the first two features as before
    
  else:
    mask = (y == 1) | (y == 2)
    X_subset = X[mask]
    y_subset = y[mask]

    #Relabel: class 1 (Versicolour) -> 0.0, class 2 (Virginica) -> 1.0
    y_processed = np.where(y_subset == 1, 0.0, 1.0)
    
    #Feature selection: Use first two features from the subset
    X_processed = X_subset[:, :2]

  #Split the dataset into training and testing sets
  X_scaled = (X_processed - X_processed.mean(axis=0)) / X_processed.std(axis=0) # Scaling features
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_processed, test_size=0.2, random_state=42)
  X_train_c = X_train.astype(np.double)
  y_train_c = y_train.astype(np.double) # y_train will now contain 0.0 and 1.0
  X_test_c = X_test.astype(np.double)
  y_test_c = y_test.astype(np.double)   # y_test will now contain 0.0 and 1.0

  if test: return X_test_c, y_test_c
  return X_train_c, y_train_c
  
#################################################################
def generate_diagnostics(test=False, lnr=True, n_patients=300, separation=3, seed=52):
  """
  This function generates synthetic diagnostic data for a binary classification problem.
  It creates a dataset for tumor classification benign as 0.0 or 1.0 as malignant.
  Args:
      test (bool): If True, returns the test set only.
      n_patients (int): Number of patients to generate.
      separation (float): Controls the separation between classes.
      seed (int or None): Random seed for reproducibility.
  Returns:
      X_train_c: Training features as a double array.
      y_train_c: Training labels as a double array.
      X_test_c: Testing features as a double array (if test=True).
      y_test_c: Testing labels as a double array (if test=True).
  """
  noise = 0.0
  if lnr is False: noise = 0.5
  X_exams, y_diagnosis = make_classification(
      n_samples=n_patients,
      n_features=2,
      n_informative=2,
      n_redundant=0,
      n_clusters_per_class=1,
      flip_y=noise,
      class_sep=separation,
      random_state=seed
  )
  X_train, X_test, y_train, y_test = train_test_split(X_exams, y_diagnosis, test_size=0.2, random_state=42)
  X_train_c = X_train.astype(np.double)
  y_train_c = y_train.astype(np.double) # y_train will now contain 0.0 and 1.0
  X_test_c = X_test.astype(np.double)
  y_test_c = y_test.astype(np.double)   # y_test will now contain 0.0 and 1.0

  if test: return X_test_c, y_test_c
  return X_train_c, y_train_c

#################################################################
def plot_data(X_train, y_train, X_test, y_test, title="Training and Testing Data", xlabel="Feature 1", ylabel="Feature 2"):
  """
  This function plots the training and testing data.
  """
  plt.figure(figsize=(10, 6))
  plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1 (Train)')
  plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0 (Train)')
  plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='cyan', label='Class 1 (Test)', marker='x')
  plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='orange', label='Class 0 (Test)', marker='x')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()

#################################################################
def plot_decision_boundary(X, y, weights, title="Decision Boundary", xlabel="Feature 1", ylabel="Feature 2"):
  """
      This function plots the decision boundary of the trained model.
  """
  plt.figure(figsize=(10, 6))
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  Z = np.dot(np.c_[np.ones_like(xx.ravel()), xx.ravel(), yy.ravel()], weights)
  Z = np.where(Z >= 0, 1, 0).reshape(xx.shape)
  plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['lightcoral', 'lightblue'], alpha=0.5)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()

#################################################################
def general_plot(data, interval, title="Model Performance Over Epochs", ylabel="Performance", xlabel="Interval"):
  """
      This function plots the performance of the model over epochs.
  """
  plt.figure(figsize=(10, 6))
  plt.plot(interval, data, marker='o')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid()
  plt.show()

#################################################################
def plot_weights(weights_history, epochs, features):
  """
      This function plots the evolution of weights during training.
  """ 
  plt.figure(figsize=(10, 6))
  for i in range(features + 1):  # +1 for the bias term
    plt.plot(epochs, [w[i] for w in weights_history], label=f'Weight {i}')
  
  plt.title('Weights Evolution Over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Weights')
  plt.legend()
  plt.grid()
  plt.show()

#################################################################