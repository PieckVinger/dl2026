import random
import math

# =========================
# Activation Functions
# =========================

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


# =========================
# Neuron Class
# =========================

class Neuron:
    def __init__(self, number_inputs):
        self.weights = []
        for i in range(number_inputs):
            self.weights.append(random.uniform(-1, 1))

        self.bias = random.uniform(-1, 1)

        self.output = 0
        self.delta = 0

    def forward(self, inputs):
        total = 0
        for i in range(len(inputs)):
            total += inputs[i] * self.weights[i]

        total += self.bias
        self.output = sigmoid(total)
        return self.output

# =========================
# Layer Class
# =========================

class Layer:
    def __init__(self, number_neurons, number_inputs):
        self.neurons = []
        for i in range(number_neurons):
            neuron = Neuron(number_inputs)
            self.neurons.append(neuron)

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))

        return outputs

# =========================
# Neural Network Class
# =========================

class NeuralNetwork:
    def __init__(self, architecture_file):
        self.layers = []

        file = open(architecture_file, "r")
        lines = file.readlines()
        file.close()

        number_layers = int(lines[0].strip())

        layer_sizes = []

        for i in range(1, number_layers + 1):
            layer_sizes.append(int(lines[i].strip()))

        for i in range(1, len(layer_sizes)):

            number_inputs = layer_sizes[i - 1]
            number_neurons = layer_sizes[i]

            layer = Layer(number_neurons, number_inputs)

            self.layers.append(layer)

    # =========================
    # Feedforward
    # =========================

    def feedforward(self, inputs):

        outputs = inputs

        for layer in self.layers:
            outputs = layer.forward(outputs)

        return outputs

    # =========================
    # Backpropagation
    # =========================

    def backpropagate(self, expected):

        # Output layer delta
        output_layer = self.layers[-1]

        for i in range(len(output_layer.neurons)):

            neuron = output_layer.neurons[i]

            error = expected[i] - neuron.output

            neuron.delta = error * sigmoid_derivative(neuron.output)

        # Hidden layers delta
        for layer_index in range(len(self.layers) - 2, -1, -1):

            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]

            for j in range(len(current_layer.neurons)):

                neuron = current_layer.neurons[j]

                error = 0

                for next_neuron in next_layer.neurons:
                    error += next_neuron.weights[j] * next_neuron.delta

                neuron.delta = error * sigmoid_derivative(neuron.output)

    # =========================
    # Update Weights
    # =========================

    def update_weights(self, inputs, learning_rate):

        current_inputs = inputs

        for layer in self.layers:

            new_inputs = []

            for neuron in layer.neurons:

                for j in range(len(current_inputs)):

                    neuron.weights[j] += (
                        learning_rate
                        * neuron.delta
                        * current_inputs[j]
                    )

                neuron.bias += learning_rate * neuron.delta

                new_inputs.append(neuron.output)

            current_inputs = new_inputs

    # =========================
    # Train Network
    # =========================

    def train(self, training_inputs,
              training_outputs,
              learning_rate,
              epochs):

        for epoch in range(epochs):

            total_error = 0

            for i in range(len(training_inputs)):

                inputs = training_inputs[i]
                expected = training_outputs[i]

                outputs = self.feedforward(inputs)

                for j in range(len(expected)):
                    total_error += (
                        expected[j] - outputs[j]
                    ) ** 2

                self.backpropagate(expected)

                self.update_weights(inputs, learning_rate)

            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch} "
                    f"Error = {total_error:.6f}"
                )

    # =========================
    # Save Weights
    # =========================

    def save_weights(self, filename):

        file = open(filename, "w")

        for layer in self.layers:

            for neuron in layer.neurons:

                line = ""

                for weight in neuron.weights:
                    line += str(weight) + " "

                line += str(neuron.bias)

                file.write(line + "\n")

        file.close()

    # =========================
    # Load Weights
    # =========================

    def load_weights(self, filename):

        file = open(filename, "r")

        lines = file.readlines()

        file.close()

        index = 0

        for layer in self.layers:

            for neuron in layer.neurons:

                values = lines[index].strip().split()

                neuron.weights = []

                for i in range(len(values) - 1):
                    neuron.weights.append(float(values[i]))

                neuron.bias = float(values[-1])

                index += 1