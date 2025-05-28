from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np


def import_data(test=False):
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
  #print("Original y:", y)
  #print("Original X:", X)

  y = np.where(y == 0.0, 1.0, 0.0)  

  #Convert to binary classification problem
  X = X[:, :2] # Using only the first two features as before
  X = (X - X.mean(axis=0)) / X.std(axis=0) # Scaling features as before

  #Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train_c = X_train.astype(np.double)
  y_train_c = y_train.astype(np.double) # y_train will now contain 0.0 and 1.0
  X_test_c = X_test.astype(np.double)
  y_test_c = y_test.astype(np.double)   # y_test will now contain 0.0 and 1.0

  """
  print("X_train", X_train)
  print("y_train", y_train)
  print("X_test", X_test)
  print("y_test", y_test)
  """

  if test: return X_test_c, y_test_c
  return X_train_c, y_train_c

#################################################################

"""
  This function plots the training and testing data.
"""
def plot_data(X_train, y_train, X_test, y_test):

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1 (Train)')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0 (Train)')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='cyan', label='Class 1 (Test)', marker='x')
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='orange', label='Class 0 (Test)', marker='x')
    plt.title('Training and Testing Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

#################################################################

"""
    This function plots the decision boundary of the trained model.
"""
def plot_decision_boundary(X, y, weights):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

    #Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    #Calculate the decision boundary
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1e10, 0], colors='lightgray', alpha=0.5)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()