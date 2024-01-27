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


def load_and_clean_data(file_path, file_path_income):
    """Load dataset and select relevant columns."""
    data = pd.read_csv(file_path, index_col='Time', parse_dates=True)
    df = pd.read_csv(file_path_income, index_col='Time', parse_dates=True)
    df = df['2021-08-30':]
    data = pd.concat([data, df], axis=1)
    data = data[['Close', 'Volume', 'Percentage Change', 'totalRevenue']]
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    data['Percentage Change'] = pd.to_numeric(data['Percentage Change'], errors='coerce')
    data['totalRevenue'] = pd.to_numeric(data['totalRevenue'], errors='coerce')
    data['totalRevenue'] = data['totalRevenue'].shift(1)
    data.dropna(how='all', inplace=True)
    data.reset_index(drop=True, inplace=True)
    features = list(data.columns)

    first_values = data.iloc[0, 0:]
    row_last = first_values

    for i, row in data.iterrows():
        for feature, value in enumerate(row):
            feature = features[feature]
            if value != value:
                value = row_last[feature]
            row_last[feature] = value
            data[feature][i] = value

    target_sensor = "Close"
    forecast_lead = 1
    target = f"{target_sensor}_pred{forecast_lead}"

    data[target] = data[target_sensor].shift(forecast_lead)
    data[target] = pd.to_numeric(data[target], errors='coerce')

    data.dropna(inplace=True)

    return data, target, features


def split_and_scale_data(data, train_ratio):
    split_index = int(len(data) * train_ratio)
    df_train = data.loc[:split_index].copy()
    df_test = data.loc[split_index:].copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    np_train = scaler.fit_transform(df_train)
    np_test = scaler.transform(df_test)

    return np_train, np_test, scaler


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


def get_loaders(data_path, data_path2):
    sequence_length = 10
    batch_size = 16

    # Data preparation
    data, target, features = load_and_clean_data(data_path, data_path2)
    scaled_train_data, scaled_test_data, scaler = split_and_scale_data(data, 0.7)

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


def get_last_sequence(data_path, data_path2):
    sequence_length = 10
    # Data preparation
    data, target, features = load_and_clean_data(data_path, data_path2)
    scaled_train_data, _, _ = split_and_scale_data(data, 1)
    train_dataset = SequenceDataset(scaled_train_data, target=target, features=features,
                                    sequence_length=sequence_length)
    last_sequence = train_dataset.X[-10:]
    last_sequence = last_sequence.reshape((-1, sequence_length, len(features)))
    return last_sequence


def create_new_sequence(last_sequence, output):
    updated_sequence = last_sequence[:, 1:, :]
    output = torch.tensor([output]).reshape(-1, 1, 1)
    volume = last_sequence[:, :, 1]
    volume_average = torch.sum(volume) / volume.size(dim=1)
    percentage_change = output[0, 0, 0] / last_sequence[0, 9, 0]
    total_revenue = last_sequence[0, 9, 3]
    volume_average, percentage_change, total_revenue = volume_average.reshape((-1, 1, 1)), percentage_change.reshape((-1, 1, 1)), total_revenue.reshape((-1, 1, 1))
    output = torch.cat([output, volume_average, percentage_change, total_revenue], dim=2)
    updated_sequence = torch.cat([updated_sequence, output], dim=1)
    return updated_sequence
