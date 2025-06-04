#include "perceptron.h"

/**
 * @brief Computes the weighted sum of inputs for a perceptron.
 *
 * @param x Pointer to the input vector.
 * @param w Pointer to the weight vector.
 * @param n Integer number of iterations
 * @return double The weighted sum of the inputs.
 * Bias is included in the x vector and supported.
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

double activation_function(double sum) {
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
    return abs(error);
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
  int *predictions = predict(x, w, samples, features);
  for (int i = 0; i < samples; i++) {
    if (predictions[i] == y[i]) {
      correct_predictions++;
    }
  }

  double accuracy = (double)correct_predictions / samples;
  free(predictions);
  return accuracy;
}

/*******************************************************************************/
/**
 * @brief Predicts the output for a set of input samples using the trained perceptron.
 *
 * @param x Pointer to the flattened input data.
 * @param w Pointer to the weight vector.
 * @param samples Number of samples in the dataset.
 * @param features Number of features (including bias) for each sample.
 * @return int* Pointer to an array containing the predicted outputs for each sample.
 */

int *predict(double *x, double *w, int samples, int features) {
  int *predictions = (int *)malloc(samples * sizeof(int));
  if (predictions == NULL) {
    return NULL; 
  }

  for (int i = 0; i < samples; i++) {
    double *current_sample_x = &x[i * features];
    predictions[i] = (int)neuron(current_sample_x, w, features);
  }
  return predictions;
}

/*******************************************************************************/
/**
 * @brief Computes the recall of the perceptron on a dataset.
 *
 * @param x Pointer to the flattened input data.
 * @param w Pointer to the weight vector.
 * @param y Pointer to the target output vector.
 * @param samples Number of samples in the dataset.
 * @param features Number of features (including bias) for each sample.
 * @return double The recall of the perceptron on the dataset.
 */

double recall(double *x, double *w, double *y, int samples, int features) {
  int true_positive = 0;
  int false_negative = 0;
  int *predictions = predict(x, w, samples, features);

  for (int i = 0; i < samples; i++) {
    if (predictions[i] == 1 && y[i] == 1) {
      true_positive++;
    } else if (predictions[i] == 0 && y[i] == 1) {
      false_negative++;
    }
  }

  free(predictions);
  
  if (true_positive + false_negative == 0) {
    return 0.0; 
  }
  
  return (double)true_positive / (true_positive + false_negative);
}
