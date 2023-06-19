from data_loader import X_train, y_train, X_val, y_val, X_test, y_test
from neural_net import NeuralNetwork
from train import train
import matplotlib.pyplot as plt

# Create neural network
net = NeuralNetwork(
    input_dim=784,
    hidden_dims=[10],
    output_dim=10,
    regularization=5e-6,
    learning_rate=1e-3
)

# Train the network
training_costs, val_costs = train(net, X_train, y_train, X_val, y_val, batch_size=200, epochs=100)

# Render the cost
fig, ax = plt.subplots()
ax.plot(training_costs, color="blue", label="train")
ax.plot(val_costs, color="orange", label="val")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average cost per epoch")
ax.set_title("Training vs validation cost")
plt.legend()
plt.show()