import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from qlstm import QLSTM
import preprocess

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
scaler = joblib.load('scaler.pkl')

# Initialize the QLSTM model
# TODO: Check input size!
input_size = 1
hidden_size = 1
n_qubits = 4
n_qlayers = 2
model = QLSTM(input_size, hidden_size, n_qubits=n_qubits, n_qlayers=n_qlayers)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)


def train_model(model, train_loader, loss_function, optimizer, n_epochs):
    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")


def test_model(model, test_loader, loss_function, scaler):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    predictions = []
    with torch.no_grad():  # No need to track gradients during testing
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            test_loss += loss.item()
            predictions.append(output)

    # Convert predictions to numpy
    predicted_points = torch.cat(predictions, dim=0).view(-1).numpy()

    # Prepare a dummy array with the correct shape
    dummy_array = np.zeros((len(predicted_points), scaler.n_features_in_))
    dummy_array[:, 0] = predicted_points  # Assuming target variable is the first feature

    # Apply inverse transform to the dummy array
    denormalized_predictions = scaler.inverse_transform(dummy_array)[:, 0].flatten()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return denormalized_predictions


n_epochs = 20

# Training the model
train_model(model, preprocess.train_loader, loss_function, optimizer, n_epochs)

# Testing the model
predicted_points = test_model(model, preprocess.test_loader, loss_function, scaler)

# Plotting
# Load the entire dataset (x and y values)
data = pd.read_csv(preprocess.data_path)
x_values = data['timesteps'].values
y_values = data['data'].values

# Calculate the starting index for test data
batch_size = preprocess.batch_size
num_train_batches = len(preprocess.train_loader)
train_data_length = batch_size * num_train_batches

# Plot the entire actual data
plt.plot(x_values, y_values, label='Actual')

# Plot the predicted points for the test data
plt.scatter(x_values[train_data_length:train_data_length + len(predicted_points)],
            predicted_points,
            color='red',
            label='Predicted',
            s=1)

plt.title('Apple Stock Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()
