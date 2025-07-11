import numpy as np

def calculate_f_xw(w, x, b):
    return np.dot(x, w) + b


def cost_func(y,f_xw):
    m = y.shape[0]
    cost = np.sum((f_xw - y)**2)/ m
    return cost

def gradient_descent(X, y, num_iterations=1000, learning_rate=0.01):
    m, n = X.shape  # m = number of samples, n = number of features
    w = np.ones(n)  # One weight per feature
    b = 1
    history = []

    for i in range(num_iterations):
        f_xw = calculate_f_xw(w, X, b)

        # Gradients
        dw = (1 / m) * np.dot(X.T, (f_xw - y))
        db = (1 / m) * np.sum(f_xw - y)

        # Update weights and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Store history
        cost = cost_func(y, f_xw)
        history.append((w.copy(), b, cost))

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return w, b, history


def predict(x,w,b):
    return calculate_f_xw(w,x,b)

# Predict on entire dataset
def predict_all(X, w, b):
    return calculate_f_xw(w,X,b)

