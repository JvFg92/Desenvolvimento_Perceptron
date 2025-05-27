#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>

int activation_function(double sum);
double neuron(double *x, double *w);
double fit(double *x, double *w, double target, double learning_rate);

#endif // PERCEPTRON_H