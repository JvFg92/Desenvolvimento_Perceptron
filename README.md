# 🧠 C-Accelerated Perceptron for Binary Classification ⚙️

This project implements a Perceptron algorithm for binary classification, with its main design functions written in C for performance optimization and wrapped in a Python class for easy use, data manipulation, and visualization. 📊 The project includes functionality to use the Iris dataset (converted to a binary problem) or generate synthetic data for training and testing.

🎯 **Goal:** Create an efficient Perceptron classifier with a C backend and a friendly Python interface.

✨ **Example View:**
<p align="center">
<img src="https://github.com/user-attachments/assets/23f1dddd-ba94-48d6-bdd1-3421bb57614e" alt="Decision Boundary Example" width="600"/>
</p>

## 🌟 Overview

Perceptron is one of the simplest supervised machine learning algorithms for binary classification. This project demonstrates:
* Implementation of the Perceptron algorithm.
* Usage of C for computationally intensive operations (neuron calculation, weight adjustment, accuracy evaluation) via `ctypes` in Python.
* A Python `Perceptron` class that encapsulates the training, prediction, evaluation, and plotting logic.
* Loading and preprocessing data for the Iris dataset and generating synthetic data. 🌸
* Splitting data into training and testing sets.
* Feature scaling (Z-score normalization).
* Training with a defined learning rate and accuracy threshold.
* Calculating error, accuracy, and recall. 📈
* K-fold cross-validation. * Visualization of:
* Training and testing data 📍
* Decision boundary of the trained model 🗺️
* Model accuracy over epochs 🎯
* Evolution of weights during training 🏋️
* Model error over epochs 📉

## ✨ Main Features

* **C Core ⚙️:** `neuron`, `fit`, `evaluate_accuracy`, `predict` and `recall` functions, innovations in C for efficiency.
* **Python Wrapper 🐍:** Easy-to-use `Perceptron` class in Python.
* **Data Sources 💾:**
* Utilize the Iris dataset (filtered for two classes and two features).
* Synthetic data generation for classification problems (linearly separable or noisy).
* **Preprocessing 🧹:**
* Convert multiclass problems to binary.
* Scale features using mean and standard deviation.
* **Training 🏋️‍♀️:**
* Iterate until a baseline accuracy is reached on the test set or a maximum number of epochs is reached.
* Store historical weights, errors, and accuracy.
* **Evaluation 📊:**
* Calculate accuracy on the training and test sets.
* Perform k-fold cross-validation.
* **Visualization 🖼️:** Use `matplotlib` to plot:
* Data distribution.
* Decision boundary.
* Learning curves (accuracy, error, weights).
* **Analysis 🖼️:** * **Flexibility 🛠️:** Allows configuration of learning rate, reference accuracy and data generation parameters.

## 📂 Project Structure

├── 📄 perceptron.c # C implementation of the main Perceptron functions

├── 📄 perceptron.h # Header file for the C code

├── 🔗 perceptron.so # Compiled shared library (generated after compilation)

├── 🐍 data_treatment.py # Functions for data import, generation and plotting

├── 🐍 training.py # Python Perceptron class and ctypes interface for C

└── 📖 main.py # Main script to run training and evaluation

## 🛠️ Prerequisites

* Python 3.12.3 🐍
* C compiler (like CGC) ⚙️
* Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn` (used in `data_treatment.py` for `load_iris` and `make_classification`)

## 🚀 Setup and Installation

1. **Clone the repository:**
```bash
git clone https://github.com/JvFg92/Perceptron_Data_Classify
cd Perceptron_Data_Classify
```

2. **Compile the C code to create a shared library (`perceptron.so`):**
On Linux or macOS:
```bash
gcc -shared -o perceptron.so -fPIC perceptron.c
```
On Windows (may require configuration depending on your compiler, e.g. with MinGW):
```bash
gcc -shared -o perceptron.so perceptron.c -Wl,--add-stdcall-alias
```
ℹ️ Make sure the resulting `perceptron.so` (or `perceptron.dll` on Windows) file is in the same directory as the Python scripts.

3. **Install Python dependencies:**
```bash
pip install numpy matplotlib scikit-learn
```

⚠️ For Linux you may need:
```bash
sudo apt install python3-numpy
sudo apt install python3-matplotlib
sudo apt install python3-sklearn
```

⚠️ For Windows you may need:
```bash
py -m pip install numpy matplotlib scikit-learn
```

✅ Ready to go!

## ▶️ Usage

The main script to run the model is `main.py`.
```bash
python main.py
```

⚠️ For Linux you may need:
```bash
python3 main.py
```
⚠️ For Windows you may need:
```bash
py main.py
```
