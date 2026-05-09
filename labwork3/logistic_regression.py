import matplotlib.pyplot as plt

# =========================
# Read CSV file manually
# =========================

x1_data = []
x2_data = []
y_data = []

file = open("loan2.csv", "r")

lines = file.readlines()

file.close()

# Skip header
for line in lines[1:]:

    values = line.strip().split(",")

    x1_data.append(float(values[0]))
    x2_data.append(float(values[1]))
    y_data.append(float(values[2]))

# =========================
# Sigmoid function
# =========================

def sigmoid(z):

    e = 2.718281828

    return 1 / (1 + (e ** (-z)))

# =========================
# Initial weights
# =========================

w0 = 0
w1 = 0
w2 = 0

# =========================
# Hyperparameters
# =========================

learning_rate = 0.001
iterations = 2000

# =========================
# Store loss values
# =========================

loss_history = []

# =========================
# Training
# =========================

for step in range(iterations):

    total_loss = 0

    for i in range(len(x1_data)):

        x1 = x1_data[i]
        x2 = x2_data[i]
        y = y_data[i]

        # Linear equation
        z = w1 * x1 + w2 * x2 + w0

        # Prediction
        y_pred = sigmoid(z)

        # Loss
        loss = 0.5 * (y_pred - y) ** 2

        total_loss += loss

        # Error
        error = y_pred - y

        # Gradients
        dw0 = error
        dw1 = error * x1
        dw2 = error * x2

        # Update weights
        w0 = w0 - learning_rate * dw0
        w1 = w1 - learning_rate * dw1
        w2 = w2 - learning_rate * dw2

    # Average loss
    average_loss = total_loss / len(x1_data)

    # Save loss
    loss_history.append(average_loss)

    # Print intermediate results
    if step % 100 == 0:

        print(f"Step {step}")
        print(f"Loss = {average_loss:.6f}")
        print(f"w0 = {w0:.6f}")
        print(f"w1 = {w1:.6f}")
        print(f"w2 = {w2:.6f}")
        print("-----------------------------")

# =========================
# Final parameters
# =========================

print("\nFinal parameters:")
print(f"w0 = {w0:.6f}")
print(f"w1 = {w1:.6f}")
print(f"w2 = {w2:.6f}")

# =========================
# Predictions
# =========================

print("\nPredictions:")

for i in range(len(x1_data)):

    x1 = x1_data[i]
    x2 = x2_data[i]

    z = w1 * x1 + w2 * x2 + w0

    probability = sigmoid(z)

    prediction = 1 if probability >= 0.5 else 0

    print(
        f"Experience={x1}, Salary={x2} -> "
        f"Probability={probability:.4f}, Prediction={prediction}"
    )

# =========================
# Plot loss graph
# =========================

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.title("Logistic Regression Training Loss")

plt.grid(True)

plt.savefig("loss.png")

plt.show()