import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# One-hot encoding the target labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# Initialize hyperparameters
input_layer_size = X.shape[1]  # 4 features in the iris dataset
hidden_layer_size = 4  # Arbitrary number of neurons in the hidden layer
output_layer_size = y_onehot.shape[1]  # 3 classes for Iris species
learning_rate = 0.1
epochs = 10000

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
np.random.seed(42)
W1 = np.random.rand(input_layer_size, hidden_layer_size)
W2 = np.random.rand(hidden_layer_size, output_layer_size)
b1 = np.zeros((1, hidden_layer_size))
b2 = np.zeros((1, output_layer_size))

# Training the Neural Network with Backpropagation
for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # Backward pass
    # Calculate error at output layer
    output_error = y_train - A2
    output_delta = output_error * sigmoid_derivative(A2)
    
    # Calculate error at hidden layer
    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(A1)
    
    # Update weights and biases
    W2 += A1.T.dot(output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += X_train.T.dot(hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(output_error))
        print(f"Epoch {epoch}/{epochs} Loss: {loss}")

# Test the Neural Network
Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

# Convert outputs to class predictions
predictions = np.argmax(A2_test, axis=1)
actual = np.argmax(y_test, axis=1)

# Accuracy
accuracy = np.mean(predictions == actual)
print(f"Accuracy: {accuracy * 100}%")
