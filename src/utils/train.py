import os

import numpy as np
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
input_size = 4
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

model_path = f'../trained_model/qlstm'
if not os.path.exists(model_path):
    os.makedirs(model_path)


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

    torch.save(model.state_dict(), model_path + '/qsltm.pth')


n_epochs = 5

best_stocks = ['NVDA', 'KO', 'MO', 'BABA', 'MA', 'V', 'JPM', 'PG', 'TSM', 'META', 'TSLA', 'GOOGL', 'AMZN',
               'MSFT', 'AAPL', 'ABBV', 'PEP', 'CRM', 'PFE', 'NFLX', 'AMD']

best_stocks = ['AAPL']

for i, stock in enumerate(best_stocks):
    data_path = f'../datasets/stock_data/{stock}.csv'
    data_path_income = f'../datasets/stock_data/{stock}_Income.csv'
    train_loader, test_loader, batch_size, scaler = preprocess.get_loaders(data_path, data_path_income)

    print(f'\n{stock} in training: {i+1}/{len(best_stocks)}')

    # Training the model
    train_model(model, train_loader, loss_function, optimizer, n_epochs)
