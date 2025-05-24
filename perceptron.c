#include "perceptron.h"

/**
 * @brief Computes the weighted sum of inputs for a perceptron.
 *
 * @param x Pointer to the input vector.
 * @param w Pointer to the weight vector.
 * @return double The weighted sum of the inputs.
 */

double perceptron(double *x, double *w, int n){
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i] * w[i];
  }
  return activation_function(sum);
}

/*******************************************************************************/
/**
 * @brief Activation function for the perceptron.
 *
 * @param sum The weighted sum of the inputs.
 * @return int The output of the perceptron (0 or 1).
 */

int activation_function(double sum) {
  return (sum >= 0) ? 1 : 0; 
}

/*******************************************************************************/
/**
 * @brief Trains the perceptron using the given input and target output.
 *
 * @param x Pointer to the input vector.
 * @param w Pointer to the weight vector.
 * @param n The number of inputs.
 * @param target The target output for training.
 * @param learning_rate The learning rate for weight updates.
 */

double train_perceptron(double *x, double *w, int n, double target, double learning_rate) {
  double output = perceptron(x, w, n);
  double error = target - output;

  for (int i = 0; i < n; i++) {
    w[i] += learning_rate * error * x[i];
  }
  return error;
}

