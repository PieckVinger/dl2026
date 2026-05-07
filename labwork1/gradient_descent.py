def f(x):
    return x ** 2

def derivative(x):
    return 2 * x

def gradient_descent(x, learning_rate, iterations):
    print(f"\nLearning rate = {learning_rate}")
    print("Step\t x\t\t f(x)")

    for step in range(iterations):
        y = f(x)

        print(f"{step}\t {x:.6f}\t {y:.6f}")

        grad = derivative(x)

        x = x - learning_rate * grad

    return x

# Experiment with different learning rates
initial_x = 10

gradient_descent(initial_x, 0.1, 20)
gradient_descent(initial_x, 0.5, 20)
gradient_descent(initial_x, 1.1, 20)