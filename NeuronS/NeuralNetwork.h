#pragma once
#include "Layer.h"
#include <memory>
class NeuralNetwork {
public:
    NeuralNetwork(double learningRate, int sizes[],int layer_count);
    double learningRate = 0.01f;
    double activation(double x);
    double derivative(double x);
    Layer* layers;
    int layer_count;
    double* feedForward(double* inputs);
    void backpropagation(double *targets);
};
