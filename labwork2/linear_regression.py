import csv

# Read CSV data
x_data = []
y_data = []

with open("lr.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        x_data.append(float(row["size"]))
        y_data.append(float(row["price"]))

# Initialize parameters
w0 = 0
w1 = 0

# Hyperparameters
learning_rate = 0.0001
iterations = 1000

n = len(x_data)

# Gradient Descent
for step in range(iterations):

    dw0 = 0
    dw1 = 0
    loss = 0

    for x, y in zip(x_data, y_data):

        y_pred = w1 * x + w0

        error = y - y_pred

        loss += error ** 2

        dw1 += (-2 / n) * x * error
        dw0 += (-2 / n) * error

    # Update weights
    w1 = w1 - learning_rate * dw1
    w0 = w0 - learning_rate * dw0

    # Mean Squared Error
    loss = loss / n

    # Print intermediate steps
    if step % 100 == 0:
        print(f"Step {step}")
        print(f"w1 = {w1:.4f}, w0 = {w0:.4f}, loss = {loss:.4f}")
        print("----------------------------------")

# Final result
print("\nFinal parameters:")
print(f"w1 = {w1:.4f}")
print(f"w0 = {w0:.4f}")