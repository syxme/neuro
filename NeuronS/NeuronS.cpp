#include <iostream>
#include "NeuralNetwork.h"




int main()
{

    int size[3] = { 2,2,1 };


    double **inputs = new double *[4];
    double **target = new double *[4];
    target[0] = new double[1]{ 0.0f };
    target[1] = new double[1]{ 1.0f };
    target[2] = new double[1]{ 1.0f };
    target[3] = new double[1]{ 0.0f };

    inputs[0] = new double[2]{ 0.0f,0.0f };
    inputs[1] = new double[2]{ 0.0f,1.0f };
    inputs[2] = new double[2]{ 1.0f,0.0f };
    inputs[3] = new double[2]{ 1.0f,1.0f };

    NeuralNetwork *nn = new NeuralNetwork(0.05f, size,3);

    int epochs = 255000;
    for (int i = 1; i < epochs; i++) {
        int right = 0;
        double errorSum = 0;
        int batchSize = 4;
        for (int j = 0; j < batchSize; j++) {
           

           double* outputs = nn->feedForward(inputs[j]);
       
           errorSum += (target[j][0] - outputs[0]) * (target[j][0] - outputs[0]);
           nn->backpropagation(target[j]);
        }
        if (i % 1000 == 0) {

        std::cout << "epoch: " << i <<". correct: " << right <<". error: " <<errorSum<<"\n";
        }

        
    }
    double *test = new double[2]{ 1,0 };
    double *outputs = nn->feedForward(test);
    std::cout << "res: " << outputs[0]<< "\n";

    test[0] = 0;
    test[1] = 0;
    outputs = nn->feedForward(test);
    std::cout << "res: " << outputs[0] << "\n";
    test[0] = 1;
    test[1] = 1;
    outputs = nn->feedForward(test);
    std::cout << "res: " << outputs[0] << "\n";  
    test[0] = 0;
    test[1] = 1;
    outputs = nn->feedForward(test);
    std::cout << "res: " << outputs[0] << "\n";
    char a;
    std::cin >> a;
}

