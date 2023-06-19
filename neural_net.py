import numpy as np
from layers import Linear, LinearAndReLU, Softmax
import pickle


"""
Define the neural network class
"""
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, regularization, learning_rate):
        """
        Initialize the layers
        """
        # Initialize linear and relu layer
        dimensions = [input_dim] + hidden_dims
        self.layers = []
        for i in range(1, len(dimensions)):
            self.layers.append(LinearAndReLU(dimensions[i-1], dimensions[i], regularization, learning_rate))

        # Initialize the final layer
        self.final_layer = Linear(output_dim, dimensions[-1], regularization, learning_rate)

        # Initialize Softmax loss
        self.softmax = Softmax()

    def update_learning_rate(self, learning_rate):
        """
        Update the learning rates of the layers
        """
        for layer in self.layers:
            layer.linear.lr = learning_rate
        self.final_layer.lr = learning_rate

    def predict(self, X):
        """
        Perform the prediction of the Neural Network
        """
        # Initialize the current value
        hidden = X

        # Pass through the layers
        for layer in self.layers:
            hidden = layer.forward(hidden)

        # Pass through the final layer
        scores = self.final_layer.forward(hidden)

        return np.argmax(scores, axis=1)

    def forward(self, X, y, training=True):
        """
        Perform the forward pass of the neural network
        """
        # Initialize the current value
        hidden = X

        # Pass through the layers
        for layer in self.layers:
            hidden = layer.forward(hidden)

        # Pass through the final layer
        scores = self.final_layer.forward(hidden)

        # Calculate the cost
        cost = self.softmax.forward(scores, y)

        if not training:
            return cost

        # Perform backward pass on final layers
        dLoss = self.softmax.backward(y)
        dLoss, _, _ = self.final_layer.backward(dLoss)

        # Perform backward pass on the layers
        for i in range(len(self.layers) - 1, -1, -1):
            dLoss, _, _ = self.layers[i].backward(dLoss)

        # Return the cost
        return cost
    
    @staticmethod
    def save(model):
        """
        Save the neural network into a pickle file
        """
        with open("model/model.pickle", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load():
        """
        Load the model from a pickle file
        """
        with open("model/model.pickle", "rb") as f:
            model = pickle.load(f)
        return model
