#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Computes the weighted sum of inputs for a perceptron.
 *
 * @param x Pointer to the input vector.
 * @param w Pointer to the weight vector.
 * @return double The weighted sum of the inputs.
 */

double perceptron(double *x, double *w){
  double sum = 0.0;
  for (int i = 0; i < 3; i++) {
    sum += x[i] * w[i];
  }
  return sum;
}

int main(){

  return 0;
}