#pragma once
class Layer
{
public:
	Layer();
	Layer(int size,int nextSize);
	int size;
	int nextSize;

	double* neurons;
	double* biases;
	double** weights;
};

