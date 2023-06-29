#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include <MyAI/NeuralNetwork.h>
#include <vector>

using MyAI::layer_t;

// -------------------------------------------- //
// YOU NEED SFML INSTALLED FOR THIS EXAMPLE !!! //
// -------------------------------------------- //

// graphic settings
const int W = 600; // screen weigth
const int H = 400; // screen height
const int pixelSize = 15;
const sf::Color color_1(0, 255, 100);
const sf::Color color_2(0, 150, 200);

// creating neural network
MyAI::NeuralNetwork network(2, 1, 2, 15);

std::vector<layer_t> points_1;
std::vector<layer_t> points_2;

void learn();
void draw(sf::CircleShape& point, sf::RectangleShape& square, sf::RenderWindow& window);

int main(int argc, char const *argv[])
{
	// creating window
	sf::RenderWindow window(sf::VideoMode(600, 400), "MyAI Example");

	// creating square
	sf::RectangleShape square;
	square.setSize(sf::Vector2f(pixelSize, pixelSize));

	// creating point
	sf::CircleShape point(8);
	point.setOutlineThickness(2);
	point.setOutlineColor(sf::Color(50, 50, 50));

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			if (event.type == sf::Event::MouseButtonPressed) {
				layer_t point {(double)event.mouseButton.x / W, (double)event.mouseButton.y / H};

				if (event.mouseButton.button == sf::Mouse::Right) { points_1.push_back(point); }
				else if (event.mouseButton.button == sf::Mouse::Left) { points_2.push_back(point); }
			}
		}
		// learn
		for (int i = 0; i < 20; ++i) {
			sf::Thread thread(&learn);
			thread.launch();
		}

		// draw
		draw(point, square, window);
	}
}

void learn() {
	for (int i = 0; i < 20; ++i) {
		for (const auto& point : points_1) {network.backPropagation(point, layer_t {1}, 0.1); }
		for (const auto& point : points_2) {network.backPropagation(point, layer_t {0}, 0.1); }
	}
}

void draw(sf::CircleShape& point, sf::RectangleShape& square, sf::RenderWindow& window) {
	layer_t inputs;
	for (int x = 0; x < W; x += pixelSize) {
		for (int y = 0; y < H; y += pixelSize) {
			square.setPosition(sf::Vector2f(x, y));
	
			inputs = {(double)x / W, (double)y / H};

			network.feedForward(inputs);

			double o = network.getOutputs()[0] * 255;
			o = fmax(fmin(o, 255), 0);
			square.setFillColor(color_1 * sf::Color(o, o, o) + color_2 * sf::Color(255-o, 255-o, 255-o));

			window.draw(square);
		}
	}

	point.setFillColor(color_1);
	for (const auto& p: points_1) {
		point.setPosition(sf::Vector2f(p[0] * W, p[1] * H));
		window.draw(point);
	}
	point.setFillColor(color_2);
	for (const auto& p: points_2) {
		point.setPosition(sf::Vector2f(p[0] * W, p[1] * H));
		window.draw(point);
	}
	
	window.display();
}