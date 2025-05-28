#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>

int activation_function(double sum);
double neuron(double *x, double *w, int n);
double fit(double *x, double *w, double target, double learning_rate, int n);
double evaluate_accuracy(double *x, double *w, double *y, int samples, int features);

#endif // PERCEPTRON_H