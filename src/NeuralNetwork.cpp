#include "MyAI/NeuralNetwork.h"
#include "MyAI/functions.h"

MyAI::NeuralNetwork::NeuralNetwork()
	: activFunction (MyAI::sigmoid), activFunctionDeriv (MyAI::sigmoidDerivative)
{}


MyAI::NeuralNetwork::NeuralNetwork(
	unsigned int inputs, unsigned int outputs,
	unsigned int layers, unsigned int neurons)
	: NeuralNetwork()
{
	this->initialize(inputs, outputs, layers, neurons);
	this->randomize(0.5);
}

// requires numbers of input & output neurons, hidden layers and neurons in each hidden layer
void MyAI::NeuralNetwork::initialize(
	unsigned int inputs, unsigned int outputs,
	unsigned int layers, unsigned int neurons)
{
	// initialize activations, weights and biases for each neuron
	weights.resize(layers + 1);
	biases.resize(layers + 1);
	activations.resize(layers + 2);

	for (int i = 0; i < layers + 1; ++i) {
		int curNum  = (i == 0) ? inputs : neurons;       // number of neurons in current layer
		int nextNum = (i == layers) ? outputs : neurons; // number of neurons in next layer

		weights[i].resize(curNum);
		for (int j = 0; j < curNum; ++j) { weights[i][j].resize(nextNum); }
		biases[i].resize(nextNum);
		activations[i].resize(curNum);
	}
	activations.back().resize(outputs);
}

void MyAI::NeuralNetwork::randomize(double scatterValue) {
	for (int i = 0; i < activations.size() - 1; ++i) {
		for (int j = 0; j < activations[i].size(); ++j) {
			for (int k = 0; k < activations[i+1].size(); ++k) {
				weights[i][j][k] += getRandomDouble(-scatterValue, scatterValue);
			}
		}
		for (int j = 0; j < activations[i+1].size(); ++j) {
			biases[i][j] += getRandomDouble(-scatterValue, scatterValue);
		}
	}
}

void MyAI::NeuralNetwork::feedForward(const MyAI::layer_t& inputs) {
	activations.front() = inputs;

	for (int i = 0; i < activations.size() - 1; ++i) {      // for every layer
		for (int j = 0; j < activations[i+1].size(); ++j) { // for every neuron in next layer
			double output = biases[i][j];

			for (int k = 0; k < activations[i].size(); ++k) { // for every neuron in current layer
				output += weights[i][k][j] * activations[i][k];
			}

			activations[i+1][j] = activFunction(output);
		}
	}
}

void MyAI::NeuralNetwork::backPropagation(const MyAI::layer_t& inputs, const MyAI::layer_t& targets, double learningRate) {
	// initialize deltas
	std::vector<MyAI::layer_t> deltas(activations.size() - 1);

	for (int i = 0; i < deltas.size(); ++i) {
		deltas[i].resize(activations[i+1].size());
	}

	// Calculate activations
	this->feedForward(inputs);

	// Calculate activations
	for (int i = 0; i < deltas.back().size(); ++i) {
		double output = activations.back()[i];
		deltas.back()[i] = activFunctionDeriv(output) * (targets[i] - output);
	}

	// Backpropagate error
	for (int i = activations.size() - 2; i > 0; --i) {
		for (int j = 0; j < activations[i].size(); ++j) { // for every neuron in current layer
			double error = 0.0;

			for (int k = 0; k < activations[i+1].size(); ++k) { // for every neuron in next layer
				error += weights[i][j][k] * deltas[i][k];
			}

			deltas[i-1][j] = error * activFunctionDeriv(activations[i][j]);
		}
	}

	// Update weights and biases
	for (int i = 0; i < activations.size() - 1; ++i) {
		for (int j = 0; j < activations[i+1].size(); ++j) {
			for (int k = 0; k < activations[i].size(); ++k) {
				weights[i][k][j] += learningRate * activations[i][k] * deltas[i][j];
			}

			biases[i][j] += learningRate * deltas[i][j];
		}
	}
}

double MyAI::NeuralNetwork::getError(const MyAI::layer_t& inputs, const MyAI::layer_t& targets) const {
	double error = 0.0;
	MyAI::layer_t outputs = this->getOutputs();

	for (int i = 0; i < outputs.size(); ++i) {
		error += std::pow(targets[i] - outputs[i], 2);
	}

	return error;
}

double MyAI::NeuralNetwork::getRandomDouble(double min, double max) const {
	return min + (double)rand() / RAND_MAX * (max - min);
}