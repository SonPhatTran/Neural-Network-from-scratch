import matplotlib.pyplot as plt
import numpy as np
from data_loader import scaler, X_train, y_train, X_val, y_val, X_test, y_test
from neural_net import NeuralNetwork
from train import train


# Create neural network
net = NeuralNetwork(
    input_dim=784,
    hidden_dims=[10],
    output_dim=10,
    regularization=5e-6,
    learning_rate=1e-3
)

# Train the network
training_costs, val_costs = train(net, X_train, y_train, X_val, y_val, batch_size=200, epochs=200)

# Check the test accuracy
y_test_predicted = net.predict(X_test)
test_acc = (y_test_predicted == y_test).mean()
print(f"Test accuracy: {test_acc:.2f}.")

# Save model
NeuralNetwork.save(net)

# Render the cost
fig, ax = plt.subplots()
ax.plot(training_costs, color="blue", label="train")
ax.plot(val_costs, color="orange", label="val")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average cost per epoch")
ax.set_title("Training vs validation cost")
plt.legend()
plt.show()

# Display the correctly classified examples
correct_examples = X_test[y_test == y_test_predicted]
correctly_classified = scaler.inverse_transform(correct_examples[np.random.randint(correct_examples.shape[0], size=4), :])
plt.subplot(221)
plt.imshow(correctly_classified[0].reshape(28, 28), cmap='gray')
plt.subplot(222)
plt.imshow(correctly_classified[1].reshape(28, 28), cmap='gray')
plt.subplot(223)
plt.imshow(correctly_classified[2].reshape(28, 28), cmap='gray')
plt.subplot(224)
plt.imshow(correctly_classified[3].reshape(28, 28), cmap='gray')
plt.show()

# Display the incorrect classified examples
incorrect_examples = X_test[y_test != y_test_predicted]
incorrectly_classified = scaler.inverse_transform(incorrect_examples[np.random.randint(incorrect_examples.shape[0], size=4), :])
plt.subplot(221)
plt.imshow(incorrectly_classified[0].reshape(28, 28), cmap='gray')
plt.subplot(222)
plt.imshow(incorrectly_classified[1].reshape(28, 28), cmap='gray')
plt.subplot(223)
plt.imshow(incorrectly_classified[2].reshape(28, 28), cmap='gray')
plt.subplot(224)
plt.imshow(incorrectly_classified[3].reshape(28, 28), cmap='gray')
plt.show()
