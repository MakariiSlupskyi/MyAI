#ifndef MY_AI_NEURAL_NETWORK_H
#define MY_AI_NEURAL_NETWORK_H

#include <vector>
#include "MyAI/functions.h"

namespace MyAI
{
	using func_t = double(*)(double); // type used for activation function and its derivative
	using layer_t = std::vector<double>;

	class NeuralNetwork
	{
	private:
		std::vector<std::vector<layer_t>> weights;
		std::vector<layer_t> biases;
		std::vector<layer_t> activations;
		
		func_t activFunction;       // reference of activation function
		func_t activFunctionDeriv;  // reference of activation function derivative

		double getRandomDouble(double min=-1, double max=1) const;

	public:
		NeuralNetwork();
		NeuralNetwork(
			unsigned int inputs, unsigned int outputs,
			unsigned int layers, unsigned int neurons);

		void setActivFunction(func_t func)      { activFunction = func; }
		void setActivFunctionDeriv(func_t func) { activFunctionDeriv = func; }
		layer_t getOutputs() const  { return activations.back(); }
		double getError(const layer_t& inputs, const layer_t& targets) const;

		void initialize(
			unsigned int inputs, unsigned int outputs,
			unsigned int layers, unsigned int neurons);
		void randomize(double scatterValue=0.2);

		void feedForward(const layer_t& inputs);
		void backPropagation(const layer_t& inputs, const layer_t& targets, double learningRate=0.01);
	};
}

#endif