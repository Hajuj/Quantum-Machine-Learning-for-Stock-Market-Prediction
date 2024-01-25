import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)


def load_and_clean_data(file_path):
    """Load dataset and select relevant columns."""
    data = pd.read_csv(file_path)
    data = data[['Time', 'Close']]
    data['Time'] = pd.to_datetime(data['Time'], errors='coerce')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    return data


def prepare_dataframe_for_lstm(df, n_steps):
    """Prepare the dataframe for LSTM by creating shifted columns."""
    df = df.copy()
    df.set_index('Time', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df


def scale_data(data):
    """Scale the data using MinMaxScaler and save the scaler."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def split_data(X, y, train_ratio=0.7):
    """Split the data into training and test sets."""
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index].copy(), X[split_index:].copy()
    y_train, y_test = y[:split_index].copy(), y[split_index:].copy()
    return X_train, X_test, y_train, y_test


def reshape_for_lstm(X, y, sequence_length):
    """Reshape the data for LSTM."""
    X = X.reshape((-1, sequence_length, 1))
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


def get_loaders(data_path):
    sequence_length = 10
    batch_size = 16

    # Data preparation
    data = load_and_clean_data(data_path)
    shifted_df = prepare_dataframe_for_lstm(data, sequence_length)
    shifted_df_as_np, scaler = scale_data(shifted_df.to_numpy())

    X, y = shifted_df_as_np[:, 1:], shifted_df_as_np[:, 0]

    X = np.flip(X, axis=1)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, y_train = reshape_for_lstm(X_train, y_train, sequence_length)
    X_test, y_test = reshape_for_lstm(X_test, y_test, sequence_length)

    # Creating datasets and data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, batch_size, scaler, sequence_length


def get_last_sequence(data_path):
    data = pd.read_csv(data_path)
    data = data[['Close']]
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    df = data.to_numpy()
    df, scaler = scale_data(df)
    last_sequence = df[-10:]
    last_sequence = last_sequence.reshape((-1, 10, 1))
    last_sequence = torch.tensor(last_sequence).float()

    return last_sequence


def create_new_sequence(last_sequence, output):
    updated_sequence = last_sequence[:, 1:, :]
    output = torch.tensor([output]).reshape(-1, 1, 1)
    updated_sequence = torch.cat([updated_sequence, output], dim=1)
    return updated_sequence
