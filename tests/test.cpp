#include <gtest/gtest.h>
#include <MyAI/NeuralNetwork.h>

using MyAI::layer_t;

// initializaion tests
TEST(InitTests, Test_1) {
	MyAI::NeuralNetwork network(2, 1, 3, 1);

	SUCCEED() << "Initialize method failed";
}
TEST(InitTests, Test_2) {
	MyAI::NeuralNetwork network;
	network.initialize(4, 6, 3, 3);

	SUCCEED() << "Initialize method failed";
}

double foo(double x) {
	return x * x;
}

// feed forward tests
TEST(FeedForwardTests, Test_1) {
	MyAI::NeuralNetwork network(2, 2, 3, 1);
	network.feedForward(layer_t {0, 1});

	SUCCEED() << "Feed forward method failed";
}
TEST(FeedForwardTests, Test_2) {
	MyAI::NeuralNetwork network(2, 2, 3, 1);
	network.setActivFunction(foo);
	network.feedForward(layer_t {0, 1});

	SUCCEED() << "Feed forward method failed";
}
TEST(FeedForwardTests, Test_3) {
	MyAI::NeuralNetwork network(2, 4, 3, 2);
	network.feedForward(layer_t {0, 1});

	SUCCEED() << "Feed forward method failed";
}

// backward propagation tests
TEST(BackPropTests, Test_1) {
	MyAI::NeuralNetwork network(2, 2, 3, 1);
	network.backPropagation(layer_t {0, 1}, layer_t {0, 1});

	SUCCEED() << "Backpropagation method failed";
}
TEST(BackPropTests, Test_2) {
	MyAI::NeuralNetwork network;
	network.initialize(2, 4, 3, 2);
	network.randomize();
	network.setActivFunction(foo);

	network.backPropagation(layer_t {0, 1}, layer_t {0, 1, 0.5, 0.9});

	SUCCEED() << "Backpropagation method failed";
}
TEST(BackPropTests, Test_3) {
	MyAI::NeuralNetwork network(2, 2, 0, 1);
	MyAI::layer_t inputs {0, 1};
	MyAI::layer_t targets {1, 1};

	double error_1 = network.getError(inputs, targets);

	for (int i = 0; i < 10; ++i) {
		network.backPropagation(inputs, targets);
	}

	double error_2 = network.getError(inputs, targets);

	ASSERT_LT(error_2, error_1) << "Backpropagation method failed";
}