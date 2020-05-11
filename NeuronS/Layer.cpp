#include "Layer.h"

Layer::Layer()
{
}

Layer::Layer(int size, int nextSize)
{
	this->size = size;
	this->nextSize = nextSize;
	neurons = new double [size];
	biases = new double [size];
	weights = new double *[size];
	for (int i = 0; i < size; i++) {
		weights[i] = new double[nextSize];
	}

}
//this.size = size;
//neurons = new double[size];
//biases = new double[size];
//weights = new double[size][nextSize];