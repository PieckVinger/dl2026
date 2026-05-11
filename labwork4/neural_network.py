import random

# =========================
# Sigmoid activation
# =========================

def sigmoid(x):

    e = 2.718281828

    return 1 / (1 + (e ** (-x)))

# =========================
# Neuron class
# =========================

class Neuron:

    def __init__(self, number_inputs):

        self.weights = []
        self.bias = 0

        # Random initialization
        for i in range(number_inputs):

            self.weights.append(random.random())

        self.bias = random.random()

    # Feedforward
    def forward(self, inputs):

        total = 0

        for i in range(len(inputs)):

            total += inputs[i] * self.weights[i]

        total += self.bias

        return sigmoid(total)

# =========================
# Layer class
# =========================

class Layer:

    def __init__(self, number_neurons, number_inputs):

        self.neurons = []

        for i in range(number_neurons):

            neuron = Neuron(number_inputs)

            self.neurons.append(neuron)

    # Feedforward for entire layer
    def forward(self, inputs):

        outputs = []

        for neuron in self.neurons:

            output = neuron.forward(inputs)

            outputs.append(output)

        return outputs

# =========================
# Neural Network class
# =========================

class NeuralNetwork:

    def __init__(self, architecture_file):

        self.layers = []

        # Read architecture file
        file = open(architecture_file, "r")

        lines = file.readlines()

        file.close()

        number_layers = int(lines[0].strip())

        layer_sizes = []

        for i in range(1, number_layers + 1):

            layer_sizes.append(int(lines[i].strip()))

        # Create layers
        for i in range(1, len(layer_sizes)):

            number_inputs = layer_sizes[i - 1]

            number_neurons = layer_sizes[i]

            layer = Layer(number_neurons, number_inputs)

            self.layers.append(layer)

    # =========================
    # Load weights from file
    # =========================

    def load_weights(self, weight_file):

        file = open(weight_file, "r")

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

    # =========================
    # Feedforward
    # =========================

    def feedforward(self, inputs):

        outputs = inputs

        for layer in self.layers:

            outputs = layer.forward(outputs)

        return outputs

# =========================
# Create network
# =========================

network = NeuralNetwork("network.txt")

# =========================
# Load XOR weights
# =========================

network.load_weights("weights.txt")

# =========================
# XOR experiments
# =========================

test_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

print("XOR Neural Network Results\n")

for inputs in test_inputs:

    output = network.feedforward(inputs)

    print(f"Input: {inputs}")
    print(f"Output: {output[0]:.6f}")
    print("-------------------------")