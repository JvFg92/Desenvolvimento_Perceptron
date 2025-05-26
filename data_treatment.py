from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

"""
  This function loads the iris dataset, converts it to a binary classification problem,
  and splits it into training and testing sets.
  Returns:
      X_train: Training features
      y_train: Training labels
      X_test: Testing features
      y_test: Testing labels
"""
def load_data(test=False):

  iris = load_iris()
  X = iris.data
  y = iris.target

  y = np.where(y == 0.0, 1.0, -1.0)  

  #Convert to binary classification problem
  X = X[:, :2]
  X = (X - X.mean(axis=0)) / X.std(axis=0)

  #Split the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train_c = X_train.astype(np.double)
  y_train_c = y_train.astype(np.double)
  X_test_c = X_test.astype(np.double)
  y_test_c = y_test.astype(np.double)

  if test: return X_test_c, y_test_c
  return X_train_c, y_train_c

#################################################################

"""_summary_
  This function generates a synthetic dataset for binary classification,
  splits it into training and testing sets, and returns them.
  Returns:
      X_train: Training features
      y_train: Training labels
      X_test: Testing features
      y_test: Testing labels
"""
def generate_linear_data(n_samples=100):
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2, 
        n_redundant=0, 
        n_informative=2, 
        n_clusters_per_class=1,
        class_sep=2.0,  
        random_state=42
    )
    y = y.astype(np.float64)
    return train_test_split(X, y, test_size=0.3, random_state=42)

#################################################################

"""
  This function plots the training and testing data.
"""
def plot_data(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1 (Train)')
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='red', label='Class -1 (Train)')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='cyan', label='Class 1 (Test)', marker='x')
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='orange', label='Class -1 (Test)', marker='x')
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

    # Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Calculate the decision boundary
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1e10, 0], colors='lightgray', alpha=0.5)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()