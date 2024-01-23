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
    data = pd.read_csv(file_path, index_col='Time', parse_dates=True)
    data = data[['Close', 'Volume']]
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    target_sensor = "Close"
    features = list(data.columns)

    forecast_lead = 1
    target = f"{target_sensor}_pred{forecast_lead}"

    data[target] = data[target_sensor].shift(-forecast_lead)
    data[target] = pd.to_numeric(data[target], errors='coerce')

    data = data.iloc[:-forecast_lead]
    return data, target, features


def split_and_scale_data(data):
    train_end = '2023-04-30'
    test_start = '2023-05-01'
    df_train = data.loc[:train_end].copy()
    df_test = data.loc[test_start:].copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    np_train = scaler.fit_transform(df_train)
    np_test = scaler.transform(df_test)
    # df_train = pd.DataFrame(np_train, columns=features)
    # df_test = pd.DataFrame(np_test, columns=features)
    # print("Test set fraction:", len(df_test) / len(data))

    return np_train, np_test, scaler


# def prepare_dataframe_for_lstm(df, n_steps):
#     """Prepare the dataframe for LSTM by creating shifted columns."""
#     df = df.copy()
#     df.set_index('Time', inplace=True)
#     for i in range(1, n_steps + 1):
#         df[f'Close(t-{i})'] = df['Close'].shift(i)
#         df[f'Volume(t-{i})'] = df['Volume'].shift(i)
#     df.dropna(inplace=True)
#     return df


# def scale_data(data):
#     """Scale the data using MinMaxScaler and save the scaler."""
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data, scaler


# def split_data(X, y, train_ratio=0.7):
#     """Split the data into training and test sets."""
#     split_index = int(len(X) * train_ratio)
#     X_train, X_test = X[:split_index].copy(), X[split_index:].copy()
#     y_train, y_test = y[:split_index].copy(), y[split_index:].copy()
#     return X_train, X_test, y_train, y_test


def reshape_for_lstm(y):
    """Reshape the data for LSTM."""
    y = y.reshape((-1, 1))
    return torch.tensor(y).float()


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[:, 0]).float()
        self.X = torch.tensor(dataframe[:, 0:len(features)]).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


def get_loaders(data_path):
    sequence_length = 10
    batch_size = 16

    # Data preparation
    data, target, features = load_and_clean_data(data_path)
    scaled_train_data, scaled_test_data, scaler = split_and_scale_data(data)

    # shifted_df = prepare_dataframe_for_lstm(data, sequence_length)
    # shifted_df_as_np, scaler = scale_data(shifted_df.to_numpy())

    # X, y = shifted_df_as_np[:, 1:], shifted_df_as_np[:, 0]

    # X = np.flip(X, axis=1)

    # X_train, X_test, y_train, y_test = split_data(X, y)

    # X_train, y_train = reshape_for_lstm(X_train, y_train, sequence_length)
    # X_test, y_test = reshape_for_lstm(X_test, y_test, sequence_length)

    # Creating datasets and data loaders
    train_dataset = SequenceDataset(scaled_train_data, target=target, features=features,
                                    sequence_length=sequence_length)
    test_dataset = SequenceDataset(scaled_test_data, target=target, features=features, sequence_length=sequence_length)

    train_dataset.y = reshape_for_lstm(train_dataset.y)
    test_dataset.y = reshape_for_lstm(test_dataset.y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    return train_loader, test_loader, batch_size, scaler


def get_last_sequence(data_path):
    data = pd.read_csv(data_path)
    data = data[['Close']]
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    _, df_test, _ = split_and_scale_data(data)
    last_sequence = df_test[-10:]
    last_sequence = last_sequence.reshape((-1, 10, 1))
    last_sequence = torch.tensor(last_sequence).float()
    return last_sequence


def create_new_sequence(last_sequence, output):
    updated_sequence = last_sequence[:, 1:, :]
    output = torch.tensor([output]).reshape(-1, 1, 1)
    updated_sequence = torch.cat([updated_sequence, output], dim=1)
    return updated_sequence
