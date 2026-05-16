import csv

from network import NeuralNetwork


# =========================
# XOR TRAINING
# =========================

print("\n=========================")
print("XOR TRAINING")
print("=========================\n")

xor_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

xor_outputs = [
    [0],
    [1],
    [1],
    [0]
]

network = NeuralNetwork("network.txt")

network.train(
    xor_inputs,
    xor_outputs,
    learning_rate=0.5,
    epochs=10000
)

print("\nXOR Results:\n")

for inputs in xor_inputs:

    output = network.feedforward(inputs)

    print(
        f"Input: {inputs} "
        f"Output: {output[0]:.6f}"
    )

network.save_weights("trained_weights.txt")


# =========================
# LOAN DATASET TRAINING
# =========================

print("\n=========================")
print("LOAN DATASET TRAINING")
print("=========================\n")

loan_inputs = []
loan_outputs = []

file = open("loan2.csv", "r")

csv_reader = csv.reader(file)

next(csv_reader)

for row in csv_reader:

    experience = float(row[0]) / 3
    salary = float(row[1]) / 10
    loan = float(row[2])

    loan_inputs.append([experience, salary])

    loan_outputs.append([loan])

file.close()

network = NeuralNetwork("network.txt")

network.train(
    loan_inputs,
    loan_outputs,
    learning_rate=0.5,
    epochs=10000
)

print("\nLoan Dataset Results:\n")

for i in range(len(loan_inputs)):

    prediction = network.feedforward(
        loan_inputs[i]
    )[0]

    print(
        f"Input: {loan_inputs[i]} "
        f"Expected: {loan_outputs[i][0]} "
        f"Predicted: {prediction:.6f}"
    )


# =========================
# HOUSE PRICE TRAINING
# =========================

print("\n=========================")
print("HOUSE PRICE TRAINING")
print("=========================\n")

house_inputs = []
house_outputs = []

file = open("lr.csv", "r")

csv_reader = csv.reader(file)

next(csv_reader)

for row in csv_reader:

    size = float(row[0]) / 80
    price = float(row[1]) / 150

    house_inputs.append([size])

    house_outputs.append([price])

file.close()

network = NeuralNetwork("network.txt")

network.train(
    house_inputs,
    house_outputs,
    learning_rate=0.5,
    epochs=10000
)

print("\nHouse Price Results:\n")

for i in range(len(house_inputs)):

    prediction = network.feedforward(
        house_inputs[i]
    )[0]

    predicted_price = prediction * 150

    real_size = house_inputs[i][0] * 80

    print(
        f"Size: {real_size:.0f} "
        f"Predicted Price: {predicted_price:.2f}"
    )