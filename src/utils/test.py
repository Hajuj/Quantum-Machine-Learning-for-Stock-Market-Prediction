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

loss_function = nn.MSELoss()

model_path = f'../trained_model/qlstm'
if not os.path.exists(model_path):
    os.makedirs(model_path)


def test_model_10day(model, last_sequence, scaler):
    model.load_state_dict(torch.load(model_path + '/qsltm.pth'))
    model.eval()
    predictions = []
    with torch.no_grad():
        output = model(last_sequence)
        new_sequence = last_sequence
        predictions.append(output)
        for _ in range(9):
            new_sequence = preprocess.create_new_sequence(new_sequence, output)
            output = model(new_sequence)
            predictions.append(output)

    predicted_points = torch.cat(predictions, dim=0).view(-1).numpy()
    dummy_array = np.zeros((len(predicted_points), scaler.n_features_in_))
    dummy_array[:, 0] = predicted_points  # Assuming target variable is the first feature

    # Apply inverse transform to the dummy array
    denormalized_predictions = scaler.inverse_transform(dummy_array)[:, 0].flatten()

    return denormalized_predictions


def test_model(model, test_loader, loss_function, scaler):
    model.load_state_dict(torch.load(model_path + '/qsltm.pth'))
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
    return denormalized_predictions, avg_test_loss


best_stocks = ['NVDA', 'DIS', 'KO', 'MO', 'BABA', 'MA', 'V', 'JPM', 'PG', 'TSM', 'META', 'TSLA', 'GOOGL', 'AMZN',
               'MSFT', 'AAPL', 'ABBV', 'PEP', 'CRM', 'PFE', 'NFLX', 'AMD', 'ABT', 'PM', 'BA', 'NKE', 'GS', 'T', 'C',
               'MU']

# best_stocks = ['NVDA']

plots = '../plots/qlstm_10'
if not os.path.exists(plots):
    os.makedirs(plots)


for i, stock in enumerate(best_stocks):
    data_path = f'../datasets/stock_data/{stock}.csv'
    train_loader, test_loader, batch_size, scaler = preprocess.get_loaders(data_path)

    print(f'Tested stock: {stock}, {i+1}/{len(best_stocks)}')

    # Testing the model
    predicted_points, avg_test_loss = test_model(model, test_loader, loss_function, scaler)
    predicted_points_np = predicted_points.tolist()

    last_sequence = preprocess.get_last_sequence(data_path)
    # torch.tensor(
    #     [[0.9600],
    #      [0.9908],
    #      [0.9814],
    #      [1.0000],
    #      [0.9763],
    #      [0.9136],
    #      [0.8656],
    #      [0.8894],
    #      [0.8679],
    #      [0.8842]])

    predicted_10_points = test_model_10day(model, last_sequence, scaler)

    # Plotting
    # Load the entire dataset (x and y values)
    data = pd.read_csv(data_path)
    data['Time'] = pd.to_datetime(data['Time'])  # Convert the 'Time' column to datetime objects

    # Convert 'Time' to the format matplotlib requires
    x_values = mdates.date2num(data['Time'].values)
    y_values = data['Close'].values

    # Calculate the starting index for test data
    num_train_batches = len(train_loader)
    train_data_length = batch_size * num_train_batches

    # difference between actual and predicted points
    x_test_area = x_values[train_data_length:train_data_length + len(predicted_points)]
    y_test_area = y_values[train_data_length:train_data_length + len(predicted_points)]  # müssen hier richtigen Bereich auswählen

    baseline_points = []
    for i in y_values[train_data_length - 1:train_data_length + len(predicted_points)]:
        baseline_points.append(i)
    baseline_points = baseline_points[:-1]

    baseline_loss = loss_function(torch.tensor(baseline_points), torch.tensor(y_test_area))

    print(f"Baseline Loss: {baseline_loss:.4f}")

    # Plot the entire actual data
    plt.plot(x_values, y_values, '-', label='Actual')

    # Plot the predicted points for the test data
    plt.plot(x_values[train_data_length:train_data_length + len(predicted_points)],
                predicted_points,
                color='red',
                label='Predicted')

    # Plot the baseline
    # plt.plot(x_values[train_data_length:train_data_length + len(predicted_points)],
    #             baseline_points,
    #             color='yellow',
    #             label='Baseline')

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

# Plot 10 days from 30-11-2023

    data = pd.read_csv(f'../datasets/stock_data_10_days/{stock}.csv')
    data['Time'] = pd.to_datetime(data['Time'])  # Convert the 'Time' column to datetime objects
    x_values = data['Time'].values
    y_values = data['Close'].values

    plt.plot(x_values, y_values, '-', label='Actual')

    plt.plot(x_values,
             predicted_10_points,
             color='red',
             label='Predicted')

    # Set the locator and formatter for the x-axis
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(f'{stock} Stock Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig(plots + f'/{stock}_10.png', dpi=300, format='png', bbox_inches='tight')

    plt.show()
