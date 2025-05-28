#include "perceptron.h"

/**
 * @brief Computes the weighted sum of inputs for a perceptron.
 *
 * @param x Pointer to the input vector.
 * @param w Pointer to the weight vector.
 * @param n Integer number of iterations
 * @return double The weighted sum of the inputs.
 */

double neuron(double *x, double *w, int n){
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
 * @param target The target output for training.
 * @param learning_rate The learning rate for weight updates.
 * @param n Integer number of iterations
 */

double fit(double *x, double *w, double target, double learning_rate, int n) {

  double output = neuron(x, w, n);
  if(output!=target){
    double error = target - output;
    for (int i = 0; i < n; i++) {
      w[i] += learning_rate * error * x[i];
    }
    return error;
  }
  return 0.0;
}

/*******************************************************************************/
/**
 * @brief Evaluates the accuracy of the perceptron on a dataset.
 *
 * @param x Pointer to the flattened input data.
 * @param w Pointer to the weight vector.
 * @param y Pointer to the target output vector.
 * @param samples Number of samples in the dataset.
 * @param features Number of features (including bias) for each sample.
 * @return double The accuracy of the perceptron on the dataset.
 */

double evaluate_accuracy(double *x, double *w, double *y, int samples, int features) {
  int correct_predictions = 0;
  for (int i = 0; i < samples; i++) {
      double *current_sample_x = &x[i * features];
      double output = neuron(current_sample_x, w, features);
      if (output == y[i]) {
        correct_predictions++;
      }
    }

  double accuracy = (double)correct_predictions / samples;
  return accuracy;
}