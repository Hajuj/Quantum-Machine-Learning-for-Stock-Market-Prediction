import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import evaluation
import preprocess
from src.models.lstm import LSTM
from src.models.qlstm import QLSTM
from src.models.qrnn import QRNN


# Model parameters
input_size = 1
hidden_size = 1
n_qubits = 4
n_qlayers = 2

# Model selection
models = {'QLSTM': QLSTM(input_size, hidden_size, n_qubits, n_qlayers),
          'QRNN': QRNN(input_size, hidden_size, n_qubits, n_qlayers),
          'LSTM': LSTM(input_size, hidden_size, 1)}

model_name = "QLSTM"
model = models[model_name]
arch = "1"

# Loss function and optimizer and scheduler
n_epochs = 50
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=(0.1 * 1 / 3), total_iters=n_epochs,
                                  verbose=True)

# Directory and trained model setup
model_path = f'../trained_model/{model_name}'
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Directory and CSV file setup
results_dir = f'../results/train/{model_name}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


def train_model(model, train_loader, loss_function, optimizer, n_epochs):
    """Train the model and return epoch and loss data."""
    small_difference_count, avg_loss = 0, 0
    epochs, loss_values = [], []

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

        current_avg_loss = total_loss / len(train_loader)
        # if abs(avg_loss - current_avg_loss) < 0.0001:
        #     small_difference_count += 1
        # else:
        #     small_difference_count = 0  # Reset counter if there's a significant change
        #
        # # Early stopping check
        # if small_difference_count >= 10:
        #     print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        #     print(f"Stopped training in epoch {epoch + 1} due to too little loss changes")
        #     break

        avg_loss = current_avg_loss

        epochs.append(epoch + 1)
        loss_values.append(avg_loss)

    return epochs, avg_loss


def save_model(model, seed, timestamp):
    """Save the trained model"""
    model_save_path = os.path.join(model_path, f"{model_name}_arch{arch}_seed{seed}_qlayer{n_qlayers}_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)

    return model_save_path


stocks = ['NVDA', 'DIS', 'KO', 'MO', 'BABA', 'MA', 'V', 'JPM', 'PG', 'TSM', 'META', 'TSLA', 'MSFT', 'AAPL', 'ABBV',
          'PEP', 'CRM', 'PFE', 'NFLX', 'AMD', 'ABT', 'PM', 'BA', 'NKE', 'GS', 'T', 'C', 'MU']

# stocks = ['NVDA']

for seed in range(1, 6):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nTraining with seed: {seed}")

    # Create CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{model_name}_seed{seed}_{timestamp}.csv"
    file_path = os.path.join(results_dir, file_name)

    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['epoch', 'stock', 'avg_loss', 'model_name', 'arch', 'n_qlayers', 'lr', 'lookback', 'batch_size', 'n_epoch',
             'seed'])

        for epoch in range(n_epochs):
            for i, stock in enumerate(stocks):
                data_path = os.path.join('..', 'datasets', 'stock_data', f'{stock}.csv')
                train_loader, test_loader, batch_size, scaler, lookback = preprocess.get_loaders(data_path)

                print(f'\n{stock} in training: {i + 1}/{len(stocks)}')

                # Training the model
                epochs, avg_loss = train_model(model, train_loader, loss_function, optimizer, n_epochs=1)

                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Seed: {seed}")

                # Save stats
                csv_writer.writerow([epoch + 1, stock, avg_loss, model_name, arch, n_qlayers, scheduler.get_last_lr()[0], lookback, batch_size, n_epochs, seed])
                # evaluation.show_loss_curve(epochs, loss_values, stock, model_name)

            scheduler.step()  # Update the scheduler

        # Save the trained model
        model_saved_path = save_model(model, seed, timestamp)
