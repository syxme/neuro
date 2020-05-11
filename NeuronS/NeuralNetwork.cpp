#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include<random>
#include<time.h>
void printDouble(double *xx, int size);
NeuralNetwork::NeuralNetwork(double learningRate, int sizes[],int layer_count)
{
    srand(time(0));
    this->learningRate = learningRate;
    this->layer_count = layer_count;
	layers = new Layer[layer_count];
    for (int i = 0; i < layer_count; i++) {
        int nextSize = 0;
        if (i < layer_count - 1) nextSize = sizes[i + 1];

        layers[i] = Layer(sizes[i], nextSize);

        for (int j = 0; j < sizes[i]; j++) {
            layers[i].biases[j] = (((rand() % 100)) * 0.01)*2.0f-1.0f ;
            std::cout << " layers[i].biases[j] : " << layers[i].biases[j] << "\n";
            for (int k = 0; k < nextSize; k++) {
                layers[i].weights[j][k] = (((rand() % 100)) * 0.01) * 2.0f - 1.0f;
            }
            printDouble(layers[i].weights[j], nextSize);
        }
    }
}


double *NeuralNetwork::feedForward(double *inputs)
{   
    memcpy(layers[0].neurons, inputs, 2*sizeof(double));

    for (int i = 1; i < layer_count; i++) {
        Layer l = layers[i - 1];
        Layer l1 = layers[i];
        for (int j = 0; j < l1.size; j++) {
            l1.neurons[j] = 0;
            for (int k = 0; k < l.size; k++) {
                l1.neurons[j] += l.neurons[k] * l.weights[k][j];
              //  std::cout << l.weights[k][j] << " ";
            }
           // std::cout << "\n";
            l1.neurons[j] -= l1.biases[j];
            l1.neurons[j] = activation(l1.neurons[j]);
        }
      //  std::cout << "=============================\n";
    }
    return layers[layer_count - 1].neurons;
}

void NeuralNetwork::backpropagation(double *targets)
{
    double* errors = new double[layers[layer_count - 1].size];
    for (int i = 0; i < layers[layer_count - 1].size; i++) {
        errors[i] = ( layers[layer_count - 1].neurons[i]- targets[i]);
    }
    for (int k = layer_count - 2; k >= 0; k--) {
        Layer left= layers[k];
        Layer right = layers[k + 1];
        double* errorsNext = new double[left.size];
        double *w_d = new double[right.size];
        double* w_err = new double[right.size];

        for (int i = 0; i < right.size; i++) {
            w_d[i] = errors[i] * derivative(right.neurons[i]);
            w_err[i] = w_d[i];
            w_d[i] *= learningRate;
        }
        double** deltas = new double*[right.size];
        for (int i = 0; i < right.size; i++) {
            deltas[i] = new double[left.size];
        }
        for (int i = 0; i < right.size; i++) {
            for (int j = 0; j < left.size; j++) {
                deltas[i][j] = w_d[i] * left.neurons[j];
            }
        }
        double **weightsNew = new double *[left.size];
        for (int i = 0; i < left.size; i++) {
            weightsNew[i] = new double[left.nextSize];
        }

        for (int i = 0; i < right.size; i++) {
            for (int j = 0; j < left.size; j++) {
                weightsNew[j][i] = left.weights[j][i] - deltas[i][j];
            }
        }

        layers[k].weights = weightsNew;
        for (int i = 0; i < left.size; i++) {
            errorsNext[i] = 0;
            for (int j = 0; j < right.size; j++) {
                errorsNext[i] += weightsNew[i][j] * w_err[j];
            }
        }
        delete []errors;
        errors = new double[left.size];
        memcpy(errors, errorsNext, left.size * sizeof(double));
        
        /*
        for (int i = 0; i < right.size; i++) {
            for (int j = 0; j < left.size; j++) {
                std::cout << weightsNew[j][i] << " ";
            }
            std::cout << "\n";
        }








        std::cout << "-------------------\n";
        for (int i = 0; i < l1.size; i++) {
            for (int j = 0; j < l.size; j++) {
                std::cout << l.weights[j][i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "-------------------\n";
        l.weights = weightsNew;
        for (int i = 0; i < l1.size; i++) {
            for (int j = 0; j < l.size; j++) {
                std::cout << l.weights[j][i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "=====================\n";
        for (int i = 0; i < l1.size; i++) {
            l1.biases[i] += w_d[i];
        }*/
        for (int i = 0; i < right.size; i++) {
            right.biases[i] += w_d[i];
        }
    }
}

void printDouble(double *xx, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << xx[i] << " ";
    }
    std::cout <<"\n";
  
}
double NeuralNetwork::activation(double x)
{
    return 1.0f / (1.0f + exp(-x));
}

double NeuralNetwork::derivative(double y)
{
    return y * (1.0f - y);
}