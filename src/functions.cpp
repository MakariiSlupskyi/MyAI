#include "MyAI/functions.h"

double MyAI::sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

double MyAI::sigmoidDerivative(double x) {
	double sigmoidValue = MyAI::sigmoid(x);
	return sigmoidValue * (1.0 - sigmoidValue);
}