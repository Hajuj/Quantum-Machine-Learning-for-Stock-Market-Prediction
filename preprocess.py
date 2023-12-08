# import numpy as np
# import pandas as pd
# from copy import deepcopy as dc
# from sklearn.preprocessing import MinMaxScaler
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
#
#
# # Load dataset
# data_path = 'datasets/dataset_500_samples_sin.csv'
# data = pd.read_csv(data_path)
#
# data = data[['x', 'y']]
#
# data['x'] = pd.to_numeric(data['x'], errors='coerce')
# data['y'] = pd.to_numeric(data['y'], errors='coerce')
#
#
# def prepare_dataframe_for_lstm(df, n_steps):
#     df = dc(df)
#
#     df.set_index('x', inplace=True)
#
#     for i in range(1, n_steps + 1):
#         df[f'y(t-{i})'] = df['y'].shift(i)
#
#     df.dropna(inplace=True)
#
#     return df
#
#
# sequence_length = 7
# shifted_df = prepare_dataframe_for_lstm(data, sequence_length)
#
# shifted_df_as_np = shifted_df.to_numpy()
#
# scaler = MinMaxScaler(feature_range=(-1, 1))
# shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
#
# X = shifted_df_as_np[:, 1:]
# y = shifted_df_as_np[:, 0]
#
# # updating recurrent data from t-7 down to t-1
# X = dc(np.flip(X, axis=1))
#
# split_index = int(len(X) * 0.7)
#
# X_train = X[:split_index]
# X_test = X[split_index:]
#
# y_train = y[:split_index]
# y_test = y[split_index:]
#
# X_train = X_train.reshape((-1, sequence_length, 1))
# X_test = X_test.reshape((-1, sequence_length, 1))
#
# y_train = y_train.reshape((-1, 1))
# y_test = y_test.reshape((-1, 1))
#
# X_train = torch.tensor(X_train).float()
# y_train = torch.tensor(y_train).float()
# X_test = torch.tensor(X_test).float()
# y_test = torch.tensor(y_test).float()
#
#
# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, i):
#         return self.X[i], self.y[i]
#
#
# train_dataset = TimeSeriesDataset(X_train, y_train)
# test_dataset = TimeSeriesDataset(X_test, y_test)
#
# batch_size = 16
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def load_and_clean_data(file_path):
    """Load dataset and select relevant columns."""
    data = pd.read_csv(file_path)
    data = data[['timesteps', 'data']]
    data['timesteps'] = pd.to_datetime(data['timesteps'], errors='coerce')
    data['data'] = pd.to_numeric(data['data'], errors='coerce')

    # print(data['timesteps'])
    return data


def prepare_dataframe_for_lstm(df, n_steps):
    """Prepare the dataframe for LSTM by creating shifted columns."""
    df = df.copy()
    df.set_index('timesteps', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'data(t-{i})'] = df['data'].shift(i)
    df.dropna(inplace=True)
    return df


def scale_data(data):
    """Scale the data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)


def split_data(X, y, train_ratio=0.7):
    """Split the data into training and test sets."""
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Make a copy of the array to ensure positive strides
    X_train = np.flip(X_train, axis=1).copy()
    X_test = np.flip(X_test, axis=1).copy()

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


data_path = 'datasets/aapl_data.csv'
sequence_length = 10
batch_size = 16

# Data preparation
data = load_and_clean_data(data_path)
shifted_df = prepare_dataframe_for_lstm(data, sequence_length)
shifted_df_as_np = scale_data(shifted_df.to_numpy())

X, y = shifted_df_as_np[:, 1:], shifted_df_as_np[:, 0]
X = np.flip(X, axis=1)

X_train, X_test, y_train, y_test = split_data(X, y)
X_train, y_train = reshape_for_lstm(X_train, y_train, sequence_length)
X_test, y_test = reshape_for_lstm(X_test, y_test, sequence_length)

# print("x_train: ", X_train.shape)
# print("x_test: ", X_test.shape)
#
# print("y_train: ", y_train.shape)
# print("y_test: ", y_test.shape)

# Creating datasets and data loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for x_batch, y_batch in train_loader:
    print("Shape of X in train_loader:", x_batch.shape)
    print("Shape of Y in train_loader:", y_batch.shape)
    break  # Only check the first batch
#
# for x_batch, y_batch in test_loader:
#     print("Shape of X in test_loader:", x_batch.shape)
#     print("Shape of Y in test_loader:", y_batch.shape)
#     break  # Only check the first batch

# # Printing the train loader
# print("Train Loader:")
# for batch_idx, (X, y) in enumerate(train_loader):
#     print(f"Batch {batch_idx}")
#     print(f"X: {X}")
#     print(f"y: {y}\n")
