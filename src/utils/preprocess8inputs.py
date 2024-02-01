import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)


def load_and_clean_data(file_path, file_path_income, file_path_cashflow, file_path_balance):
    """Load dataset and select relevant columns."""
    data = pd.read_csv(file_path, index_col='Time', parse_dates=True)
    data_income = pd.read_csv(file_path_income, index_col='Time', parse_dates=True)
    data_cashflow = pd.read_csv(file_path_cashflow, index_col='Time', parse_dates=True)
    data_balance = pd.read_csv(file_path_balance, index_col='Time', parse_dates=True)
    data_income, data_cashflow, data_balance = data_income['2021-08-30':], data_cashflow['2021-08-30':], data_balance['2021-08-30':]
    data = pd.concat([data, data_income, data_cashflow, data_balance], axis=1)
    data = data[['Close', 'Volume', 'Percentage Change', 'totalRevenue', 'ebit', 'operatingCashflow', 'totalAssets', 'longTermDebt']]
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    data['Percentage Change'] = pd.to_numeric(data['Percentage Change'], errors='coerce')
    data['totalRevenue'] = pd.to_numeric(data['totalRevenue'], errors='coerce')
    data['ebit'] = pd.to_numeric(data['ebit'], errors='coerce')
    data['operatingCashflow'] = pd.to_numeric(data['operatingCashflow'], errors='coerce')
    data['totalAssets'] = pd.to_numeric(data['totalAssets'], errors='coerce')
    data['longTermDebt'] = pd.to_numeric(data['longTermDebt'], errors='coerce')
    data['totalRevenue'] = data['totalRevenue'].shift(1)
    data['ebit'] = data['ebit'].shift(1)
    data['operatingCashflow'] = data['operatingCashflow'].shift(1)
    data['totalAssets'] = data['totalAssets'].shift(1)
    data['longTermDebt'] = data['longTermDebt'].shift(1)
    data.dropna(how='all', inplace=True)
    data.reset_index(drop=True, inplace=True)
    features = list(data.columns)

    first_values = data.iloc[0, :]
    row_last = first_values

    for i, row in data.iterrows():
        for feature, value in enumerate(row):
            feature = features[feature]
            if value != value:
                value = row_last[feature]
                data[feature][i] = value
            row_last[feature] = value

    data['longTermDebt'] = data['longTermDebt'].fillna(0)

    return data, features


def split_and_scale_data(data, train_ratio):
    split_index = int(len(data) * train_ratio)
    train = data[:split_index].copy()
    test = data[split_index:].copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return train, test, scaler


def create_sequences(data, sequence_length):
    """Prepare the dataframe for LSTM by creating shifted columns."""
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]
        y = data[i + sequence_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def reshape_for_lstm(X, y, sequence_length, features):
    """Reshape the data for LSTM."""
    X = X.reshape((-1, sequence_length, len(features)))
    y = y.reshape((-1, 1))
    return torch.tensor(X).float(), torch.tensor(y).float()


class TimeSeriesDataset(Dataset):
    """Custom Dataset for Time Series data."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(sequence_length, batch_size, train_ratio, data_path, data_path2, data_path3, data_path4):
    # Data preparation
    data, features = load_and_clean_data(data_path, data_path2, data_path3, data_path4)
    train, test, scaler = split_and_scale_data(data, train_ratio)
    X_train, y_train = create_sequences(train, sequence_length)
    X_test, y_test = create_sequences(test, sequence_length)

    # getting back first 10 values
    X_test = np.append(X_train[-sequence_length:], X_test, axis=0)
    y_test = np.append(test[:sequence_length, 0], y_test, axis=0)

    # reshape
    X_train, y_train = reshape_for_lstm(X_train, y_train, sequence_length, features)
    X_test, y_test = reshape_for_lstm(X_test, y_test, sequence_length, features)

    # Creating datasets and data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler


def get_last_sequence(sequence_length, train_ratio, data_path, data_path2, data_path3, data_path4):
    # Data preparation
    data, features = load_and_clean_data(data_path, data_path2, data_path3, data_path4)
    train, _, _ = split_and_scale_data(data, train_ratio)
    last_sequence = train[-sequence_length:]
    last_sequence = last_sequence.reshape((-1, sequence_length, len(features)))
    last_sequence = torch.tensor(last_sequence).float()
    return last_sequence


def update_recurrent_sequence(sequence_length, last_sequence, output):
    updated_sequence = last_sequence[:, 1:, :]
    output = torch.tensor([output]).reshape(-1, 1, 1)
    volume = last_sequence[:, :, 1]
    volume_average = torch.sum(volume) / volume.size(dim=1)
    percentage_change = output[0, 0, 0] / last_sequence[0, sequence_length-1, 0]
    total_revenue = last_sequence[0, sequence_length-1, 3]
    ebit = last_sequence[0, sequence_length-1, 4]
    cashflow = last_sequence[0, sequence_length-1, 5]
    assets = last_sequence[0, sequence_length-1, 6]
    debts = last_sequence[0, sequence_length-1, 7]
    volume_average, percentage_change, total_revenue = volume_average.reshape((-1, 1, 1)), percentage_change.reshape((-1, 1, 1)), total_revenue.reshape((-1, 1, 1))
    ebit, cashflow, assets, debts = ebit.reshape((-1, 1, 1)), cashflow.reshape((-1, 1, 1)), assets.reshape((-1, 1, 1)), debts.reshape((-1, 1, 1))
    output = torch.cat([output, volume_average, percentage_change, total_revenue, ebit, cashflow, assets, debts], dim=2)
    updated_sequence = torch.cat([updated_sequence, output], dim=1)
    return updated_sequence