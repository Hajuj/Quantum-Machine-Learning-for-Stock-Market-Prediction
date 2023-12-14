import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import preprocess
from src.models.lstm import LSTM
from src.models.qlstm import QLSTM
from src.models.qrnn import QRNN

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize the model
input_size = 1
hidden_size = 1
n_qubits = 4
n_qlayers = 2

QLSTM = QLSTM(input_size, hidden_size, n_qubits=n_qubits, n_qlayers=n_qlayers)
QRNN = QRNN(input_size, hidden_size, n_qubits=n_qubits, n_qlayers=n_qlayers)
LSTM = LSTM(input_size, hidden_size, 1)

model = QLSTM

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


# def test_model(model, test_loader, loss_function, scaler):
#     model.eval()  # Set the model to evaluation mode
#     test_loss = 0
#     all_predictions = []
#
#     with torch.no_grad():  # No need to track gradients during testing
#         for X_batch, y_batch in test_loader:
#             updated_X_batch = X_batch.clone()  # Create a copy to update
#             for i in range(updated_X_batch.shape[0]):  # Loop over the batch size
#                 for j in range(updated_X_batch.shape[1]):  # Loop over the sequence length
#                     if updated_X_batch[i, j, 0] == -1.0000:
#                         # Predict and update the value
#                         output = model(updated_X_batch)
#                         updated_X_batch[i, j, 0] = output[i]
#
#             # Final prediction with the updated X_batch
#             final_output = model(updated_X_batch)
#             loss = loss_function(final_output, y_batch)
#             test_loss += loss.item()
#             all_predictions.append(final_output)
#
#     # Convert predictions to numpy
#     predicted_points = torch.cat(all_predictions, dim=0).view(-1).numpy()
#
#     # Prepare a dummy array with the correct shape
#     dummy_array = np.zeros((len(predicted_points), scaler.n_features_in_))
#     dummy_array[:, 0] = predicted_points  # Assuming target variable is the first feature
#
#     # Apply inverse transform to the dummy array
#     denormalized_predictions = scaler.inverse_transform(dummy_array)[:, 0].flatten()
#
#     avg_test_loss = test_loss / len(test_loader)
#     print(f"Test Loss: {avg_test_loss:.4f}")
#     return denormalized_predictions

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

best_stocks = ['NVDA', 'DIS', 'KO', 'MO', 'BABA', 'MA', 'V', 'JPM', 'PG', 'TSM', 'META', 'TSLA', 'GOOGL', 'AMZN',
               'MSFT', 'AAPL', 'ABBV', 'PEP', 'CRM', 'PFE', 'NFLX', 'AMD', 'ABT', 'PM', 'BA', 'NKE', 'GS', 'T', 'C',
               'MU']

plots = '../plots/lstm_alex'
if not os.path.exists(plots):
    os.makedirs(plots)

for stock in best_stocks:
    data_path = f'../datasets/stock_data/{stock}.csv'
    train_loader, test_loader, batch_size, scaler = preprocess.get_loaders(data_path)

    # Training the model
    train_model(model, train_loader, loss_function, optimizer, n_epochs)

    # Testing the model
    predicted_points = test_model(model, test_loader, loss_function, scaler)

    # Plotting
    # Load the entire dataset (x and y values)
    data = pd.read_csv(data_path)
    data['Time'] = pd.to_datetime(data['Time'])  # Convert the 'Time' column to datetime objects
    y_values = data['Close'].values

    # Convert 'Time' to the format matplotlib requires
    x_values = mdates.date2num(data['Time'].values)

    # Calculate the starting index for test data
    num_train_batches = len(train_loader)
    train_data_length = batch_size * num_train_batches

    # Plot the entire actual data
    plt.plot(x_values, y_values, '-', label='Actual')

    # Plot the predicted points for the test data
    plt.scatter(x_values[train_data_length:train_data_length + len(predicted_points)],
                predicted_points,
                color='red',
                label='Predicted',
                s=3)

    # Set the locator and formatter for the x-axis
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(f'{stock} Stock Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig(plots + f'/{stock}.png', dpi=300, format='png', bbox_inches='tight')

    plt.show()
