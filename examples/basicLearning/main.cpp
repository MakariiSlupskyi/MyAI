#include <iostream>
#include <MyAI/NeuralNetwork.h>

const int inputNum  = 2; // number of network inputs
const int ouputNum  = 3; // number of network outputs
const int layerNum  = 1; // number of hidden layers
const int neuronNum = 3; // numder of neurons in each hidden layer

int main(int argc, char const *argv[])
{
	// creating neural network
	MyAI::NeuralNetwork network;
	network.initialize(inputNum, ouputNum, layerNum, neuronNum);
	network.randomize(0.5);

	// learning network
	MyAI::layer_t inputs  { 0, 1 };
	MyAI::layer_t targets { 1, 0, 1 };

	for (int i = 0; i < 10000; ++i) {
		network.backPropagation(inputs, targets);

		if (i % 100 == 0) {
			std::cout << "iteration: " << i << ". Error: " << network.getError(inputs, targets) << std::endl;
		}
	}

	std::cin.get();

	return 0;
}