import numpy as np


def train(network, X_train, y_train, X_val, y_val,
          batch_size, epochs, report_after=10, decay_after=50, 
          learning_rate = 1e-3, learning_rate_decay=0.99):
    """
    Train a neural network
    Parameters:
    - network: the neural network to train
    - X_train: the train features of size (N x D)
    - y_train: the train labels of size (N,)
    - X_val: the validation features of size (N x D)
    - y_val: the validation labels of size (N,)
    - batch_size: the size of one training batch
    - epochs: the number of epochs
    Returns:
    - The costs of each epochs
    """
    # Initialize the costs array
    training_costs = []
    val_costs = []

    for epoch in range(epochs):
        # Epoch cost
        epoch_costs = []

        # Shuffle the training set
        random_indices = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=False)
        X_shuffled = X_train[random_indices]
        y_shuffled = y_train[random_indices]

        # Get the batches
        X_batches = np.array_split(X_shuffled, batch_size)
        y_batches = np.array_split(y_shuffled, batch_size)

        # Forward pass and Backward pass
        for X_batch, y_batch in zip(X_batches, y_batches):
            # Perform the forward pass
            cost = network.forward(X_batch, y_batch, training=True)
            # Record the cosst
            epoch_costs.append(cost)

        # Calculate the average training cost
        avg_epoch_cost = np.mean(epoch_costs)
        training_costs.append(avg_epoch_cost)

        # Calculate the validation cost
        val_cost = network.forward(X_val, y_val, training=False)
        val_costs.append(val_cost)

        # Display the cost
        if (epoch + 1) % report_after == 0:
            # Print the train and validation accuracy
            y_train_predicted = network.predict(X_train)
            train_acc = (y_train_predicted == y_train).mean()
            y_val_predicted = network.predict(X_val)
            val_acc = (y_val_predicted == y_val).mean()
            print("---------------------------------------------------------------------------")
            print(f"Epoch {epoch}: average cost {avg_epoch_cost:.2f}, train accuracy {train_acc:.2f}, val accuracy {val_acc:.2f}.")

        # Learning rate update
        if (epoch + 1) % decay_after == 0:
            learning_rate = learning_rate * learning_rate_decay
            network.update_learning_rate(learning_rate)

    return training_costs, val_costs