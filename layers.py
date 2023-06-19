import numpy as np


"""
Define a class for the linear layer
"""
class Linear:
    def __init__(self, in_features, out_features, regularization=0.0, learning_rate=3e-3):
        """
        Initialize the model's parameters
        """
        # Create the weight (with Xavier initialization) and bias
        self.W = np.random.randn(in_features, out_features) / 10 # * np.sqrt(2 / in_features)
        self.b = np.zeros((out_features,))
        self.reg = regularization
        self.lr = learning_rate

    def forward(self, X):
        """
        Perform the forward pass of the model
        Cache the result for the backward pass
        """
        self.cache = (X,)
        result = X @ self.W + self.b
        return result

    def backward(self, dLoss):
        """
        Perform the backward pass of the model
        Calculate the gradients and updates the weight and bias
        """
        # Get the previous input
        X = self.cache[0]

        # Calculate the gradients
        dW = X.T @ dLoss + self.reg * self.W
        db = np.sum(dLoss, axis=0) + self.reg * self.b
        dX = dLoss @ self.W.T

        # Update the weight and bias
        self.W -= self.lr * dW
        self.b -= self.lr * db

        # Return the gradients
        return dX, dW, db
    

"""
Define a class for the ReLU layer
"""
class ReLU:
    def forward(self, X):
        """
        Perform the forward pass of the ReLU layer
        Cache the result for the backward pass
        """
        self.cache = (X,)
        result = np.maximum(X, 0)
        return result

    def backward(self, dLoss):
        """
        Perform the backward pass of the ReLU layer
        """
        # Get the input from cache
        X = self.cache[0]

        # Calculate the gradient
        mask = (X > 0).astype(np.float32)
        dX = dLoss * mask

        # Return the gradient
        return dX
    

"""
Define a layer that combines the Linear layer and the ReLU layer
"""
class LinearAndReLU:
    def __init__(self, in_features, out_features, regularization, learning_rate):
        """
        Create two layers
        """
        self.linear = Linear(in_features, out_features, regularization, learning_rate)
        self.relu = ReLU()

    def forward(self, X):
        """
        Perform the forward pass on the two layers
        """
        hidden = self.linear.forward(X)
        return self.relu.forward(hidden)

    def backward(self, dLoss):
        """
        Perform the backward pass of the two layers
        """
        dHidden = self.relu.backward(dLoss)
        dX, dW, db = self.linear.backward(dHidden)
        return dX, dW, db


"""
Define a Softmax layer
"""
class Softmax:
    def forward(self, scores, y):
        """
        Perform the forward pass of the Softmax layer
        Cache the result for the backward pass
        """
        # Get the dimension
        N, C = scores.shape

        # Exponential
        E = np.exp(scores)

        # Distribution
        D = E / np.sum(E, axis=1).reshape(-1, 1)

        # Choose the probabability
        P = D[np.arange(N), y]
        self.cache = (D,)

        # Calculate the log probability
        L = -np.log(P)

        # Calculate the cost
        cost = (1 / N) * np.sum(L)
        return cost

    def backward(self, y):
        """
        Perform the backward pass of the Softmax layer
        """
        # Get the cache
        D = self.cache[0]
        
        # Get the dimension
        N, C = D.shape

        # Calculate the gradient
        dScores = D
        dScores[np.arange(N), y] -= 1
        dScores = (1 / N) * dScores

        # Return the gradient
        return dScores