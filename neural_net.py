from layers import Linear, ReLU, Softmax


"""
Define the neural network class
"""
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, regularization):
        """
        Initialize the layers
        """
        # Initialize linear and relu layer
        dimensions = [input_dim] + hidden_dims
        self.layers = []
        for i in range(1, len(dimensions)):
            self.layers.append(Linear(dimensions[i-1], dimensions[i], regularization))

        # Initialize the final layer
        self.final_layers = Linear(output_dim, dimensions[-1], regularization)

        