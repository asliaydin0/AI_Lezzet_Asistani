import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(test_input, X_train, y_train, lr=0.1, epochs=500):
    input_size = X_train.shape[1]
    weights = np.random.randn(input_size, 1)
    bias = 0

    for _ in range(epochs):
        z = np.dot(X_train, weights) + bias
        y_hat = sigmoid(z)
        error = y_hat - y_train
        dw = np.dot(X_train.T, error) / len(X_train)
        db = np.mean(error)
        weights -= lr * dw
        bias -= lr * db

    # Test verisi tahmini
    z_test = np.dot(test_input, weights) + bias
    return sigmoid(z_test)[0][0]
