import csv
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import evaluation
import plot
import preprocess
import preprocess8inputs
import test
import train
from src.models import baseline
from src.models.lstm import LSTM
from src.models.qlstm import QLSTM
from src.models.qrnn import QRNN

# Training parameters
train_ratio = 0.7
sequence_length = 10
batch_size = 16
n_epochs = 50
lr = 0.03

# Model parameters
hidden_size = 1

arch_options = {
    '1.1': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 4, 'n_qlayers': 2, 'input_size': 4},
    '1.2': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 4, 'n_qlayers': 3, 'input_size': 4},
    '1.3': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 8, 'n_qlayers': 2, 'input_size': 4},
    '1.4': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 8, 'n_qlayers': 3, 'input_size': 4},

    '2.1': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 4, 'n_qlayers': 2, 'input_size': 4},
    '2.2': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 4, 'n_qlayers': 3, 'input_size': 4},
    '2.3': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 8, 'n_qlayers': 2, 'input_size': 4},
    '2.4': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 8, 'n_qlayers': 3, 'input_size': 4},

    '3.1': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 4, 'n_qlayers': 2, 'input_size': 8},
    '3.2': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 4, 'n_qlayers': 3, 'input_size': 8},
    '3.3': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 8, 'n_qlayers': 2, 'input_size': 8},
    '3.4': {'variational_layer': qml.templates.BasicEntanglerLayers, 'n_qubits': 8, 'n_qlayers': 3, 'input_size': 8},

    '4.1': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 4, 'n_qlayers': 2, 'input_size': 8},
    '4.2': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 4, 'n_qlayers': 3, 'input_size': 8},
    '4.3': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 8, 'n_qlayers': 2, 'input_size': 8},
    '4.4': {'variational_layer': qml.templates.StronglyEntanglingLayers, 'n_qubits': 8, 'n_qlayers': 3, 'input_size': 8},
        }

# Model selection
models = {'QLSTM': lambda config: QLSTM(input_size=config['input_size'], hidden_size=hidden_size, n_qubits=config['n_qubits'], n_qlayers=config['n_qlayers'], variational_layer=config['variational_layer']),
          'QRNN': lambda config: QRNN(input_size=config['input_size'], hidden_size=hidden_size, n_qubits=config['n_qubits'], n_qlayers=config['n_qlayers']),
          'LSTM': lambda config: LSTM(input_size=config['input_size'], hidden_size=hidden_size, num_stacked_layers=1)}

arch = '1.4'
config = arch_options[arch]

# Create model using the selected architecture
model = models['QLSTM'](config)
model_name = "QLSTM"
quantum = True

n_qubits = config['n_qubits']
n_qlayers = config['n_qlayers']

# Loss function and optimizer and scheduler
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=(0.1 * 1 / 3), total_iters=n_epochs,
                                  verbose=True)

# Directory and trained model setup
model_path = f'../trained_model/{model_name}'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Directory and CSV file setup
results_train_dir = f'../results/train/{model_name}'
if not os.path.exists(results_train_dir):
    os.makedirs(results_train_dir)

results_test_dir = f'../results/test/{model_name}'
if not os.path.exists(results_test_dir):
    os.makedirs(results_test_dir)

stocks = ['NVDA', 'DIS', 'KO', 'MO', 'BABA', 'MA', 'V', 'JPM', 'PG', 'TSM', 'META', 'TSLA', 'MSFT', 'AAPL', 'ABBV',
          'PEP', 'CRM', 'PFE', 'NFLX', 'AMD', 'ABT', 'PM', 'BA', 'NKE', 'GS', 'T', 'C', 'MU']
# stocks = ['AAPL', 'KO']

# Heatmap data
selected_stocks = ['AAPL', 'KO', 'BABA', 'MA', 'PG', 'PFE', 'NKE', 'TSLA', 'T', 'PM']
selected_stocks_with_result_file = []
# selected_stocks = ['AAPL', 'KO']


def save_model(model, seed, timestamp, lookback):
    """Save the trained model"""
    model_save_path = os.path.join(model_path, f"{model_name}_arch{arch}_seed{seed}_lookback{lookback}_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved in {model_save_path}')

    return model_save_path


for seed in range(1, 6):
    # Training
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.set_seed(seed)

    print(f"\nTraining with seed: {seed}")

    # Create CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{model_name}_seed{seed}_{timestamp}.csv"
    train_file_path = os.path.join(results_train_dir, file_name)

    with open(train_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['epoch', 'stock', 'avg_loss', 'model_name', 'arch', 'n_qubits', 'n_qlayers', 'lr', 'lookback', 'batch_size', 'n_epoch',
             'seed'])

        for epoch in range(n_epochs):
            for i, stock in enumerate(stocks):
                data_path = os.path.join('..', 'datasets', 'stock_data', f'{stock}.csv')
                data_path_income = os.path.join('..', 'datasets', 'stock_data', f'{stock}_Income.csv')
                data_path_cashflow = os.path.join('..', 'datasets', 'stock_data', f'{stock}_Cashflow.csv')
                data_path_balance = os.path.join('..', 'datasets', 'stock_data', f'{stock}_Balance.csv')

                if arch == "1.1" or arch == "1.2" or arch == "1.3" or arch == "1.4" or arch == "2.1" or arch == "2.2" or arch == "2.3" or arch == "2.4":
                    train_loader, test_loader, scaler = preprocess.get_loaders(sequence_length, batch_size, train_ratio, data_path, data_path_income)
                elif arch == "3.1" or arch == "3.2" or arch == "3.3" or arch == "3.4" or arch == "4.1" or arch == "4.2" or arch == "4.3" or arch == "4.4":
                    train_loader, test_loader, scaler = preprocess8inputs.get_loaders(sequence_length, batch_size, train_ratio, data_path, data_path_income, data_path_cashflow, data_path_balance)
                else:
                    print("Invalid architecture during training!!")
                    exit()

                print(f'\n{stock} in training: {i + 1}/{len(stocks)}')

                # Training the model
                epochs, avg_loss = train.train_model(model, train_loader, loss_function, optimizer, n_epochs=1)

                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Seed: {seed}")

                # Save stats
                csv_writer.writerow(
                    [epoch + 1, stock, avg_loss, model_name, arch, n_qubits, n_qlayers, format(scheduler.get_last_lr()[0], '.6f'),
                     sequence_length, batch_size, len(epochs), seed])

            scheduler.step()  # Update the scheduler

        # Save the trained model
        model_saved_path = save_model(model, seed, timestamp, sequence_length)

    # Testing
    for i, stock in enumerate(stocks):
        data_path = f'../datasets/stock_data/{stock}.csv'
        data_path_income = os.path.join('..', 'datasets', 'stock_data', f'{stock}_Income.csv')
        data_path_cashflow = os.path.join('..', 'datasets', 'stock_data', f'{stock}_Cashflow.csv')
        data_path_balance = os.path.join('..', 'datasets', 'stock_data', f'{stock}_Balance.csv')

        if arch == "1.1" or arch == "1.2" or arch == "1.3" or arch == "1.4" or arch == "2.1" or arch == "2.2" or arch == "2.3" or arch == "2.4":
            train_loader, test_loader, scaler = preprocess.get_loaders(sequence_length, batch_size, train_ratio,
                                                                       data_path, data_path_income)
        elif arch == "3.1" or arch == "3.2" or arch == "3.3" or arch == "3.4" or arch == "4.1" or arch == "4.2" or arch == "4.3" or arch == "4.4":
            train_loader, test_loader, scaler = preprocess8inputs.get_loaders(sequence_length, batch_size, train_ratio,
                                                                              data_path, data_path_income,
                                                                              data_path_cashflow, data_path_balance)
        else:
            print("Invalid architecture while testing!!")
            exit()

        stock_plot_path = f'../plots/{model_name}/{stock}'
        if not os.path.exists(stock_plot_path):
            os.makedirs(stock_plot_path)

        evaluation_plot_path = f'../plots/evaluation/{model_name}/{stock}'
        if not os.path.exists(evaluation_plot_path):
            os.makedirs(evaluation_plot_path)

        results = results_test_dir + f'/{stock}'
        if not os.path.exists(results):
            os.makedirs(results)

        print(f'\nTested stock: {stock}, {i + 1}/{len(stocks)}')

        # Testing the model
        predicted_points, avg_test_loss = test.test_model(model, test_loader, loss_function, scaler, model_saved_path)

        if arch == "1.1" or arch == "1.2" or arch == "1.3" or arch == "1.4" or arch == "2.1" or arch == "2.2" or arch == "2.3" or arch == "2.4":
            last_sequence = preprocess.get_last_sequence(sequence_length, train_ratio, data_path, data_path_income)
        elif arch == "3.1" or arch == "3.2" or arch == "3.3" or arch == "3.4" or arch == "4.1" or arch == "4.2" or arch == "4.3" or arch == "4.4":
            last_sequence = preprocess8inputs.get_last_sequence(sequence_length, train_ratio, data_path, data_path_income, data_path_cashflow, data_path_balance)
        else:
            print("Invalid architecture while getting last sequence!!")
            exit()

        predicted_10_points = test.test_model_10day(model, last_sequence, scaler, model_saved_path, sequence_length, arch)

        # Plotting
        data = pd.read_csv(data_path)
        data['Time'] = pd.to_datetime(data['Time'])  # Convert the 'Time' column to datetime objects

        # Convert 'Time' to the format matplotlib requires
        x_values = data['Time'].values
        y_values = data['Close'].values
        percentage_changes = data['Percentage Change']

        # Calculate the starting index for test data        TODO : Index check
        num_train_batches = len(train_loader)
        train_data_length = batch_size * num_train_batches

        # difference between actual and predicted points
        x_test_area = x_values[train_data_length:train_data_length + len(predicted_points)]
        y_test_area = y_values[train_data_length:train_data_length + len(predicted_points)]

        # plot.plot_1_day_predictions(predicted_points, y_test_area, x_test_area, stock, stock_plot_path)

        # Save test data to csv

        # Calculate Accuracy Score
        last_actual_value = y_values[train_data_length - 1]
        test_percentage_changes = percentage_changes[train_data_length:train_data_length + len(predicted_points)]
        accuracy_score = evaluation.calculate_accuracy_score(test_percentage_changes, predicted_points,
                                                             last_actual_value)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"1_seed{seed}_arch{arch}_qubits{n_qubits}_qlayers{n_qlayers}_lookback{sequence_length}_batch{batch_size}_{timestamp}.csv"
        test_file_path = os.path.join(results, file_name)

        if selected_stocks.__contains__(stock):
            selected_stocks_with_result_file.append([stock, test_file_path])

        constants = [model_name, arch, n_qubits, n_qlayers, seed, sequence_length, batch_size]
        evaluation.save_data_to_csv(predicted_points, y_test_area, x_test_area, accuracy_score, stock, constants,
                                    test_file_path)

        # Plot 10 days from 30-11-2023
        data = pd.read_csv(f'../datasets/stock_data/{stock}.csv')
        data['Time'] = pd.to_datetime(data['Time'])  # Convert the 'Time' column to datetime objects
        data10 = data[int(len(data) * train_ratio):int(len(data) * train_ratio) + 10].copy()
        x_values = data10['Time'].values
        y_values = data10['Close'].values

        # plot.plot_10_day_prediction(predicted_10_points, x_values, y_values, stock, stock_plot_path)

        file_name = f"10_seed{seed}_arch{arch}_qubits{n_qubits}_qlayers{n_qlayers}_lookback{sequence_length}_batch{batch_size}_{timestamp}.csv"
        test_file_path_10_day = os.path.join(results, file_name)
        evaluation.save_data_to_csv_no_accuracy(predicted_10_points, y_values, x_values, stock, constants, test_file_path_10_day)

        # Save 10 day prediction to csv

        # Baseline using Linear Regression
        baseline_points = baseline.get_baseline_points(test_loader, scaler)

        # Plotting the Baseline
        # plot.plot_baseline(baseline_points, y_test_area, x_test_area, stock, stock_plot_path)

        # Evaluation

        # Loss Curve plotting
        # plot.plot_loss_curve(train_file_path, evaluation_plot_path, stock, seed)

    # accumulated_evaluation_path = f'../plots/evaluation/{model_name}'
    # plot.plot_accumulated_loss_curve(train_file_path, accumulated_evaluation_path, seed)

    # plot.plot_heatmap(selected_stocks_with_result_file, accumulated_evaluation_path)
