import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Function to create sequences from x and y values
def create_sequences(x_values, y_values, seq_length):
    x_sequences = []
    y_sequences = []
    for i in range(len(x_values) - seq_length + 1):
        x_seq = x_values[i:i + seq_length]
        y_seq = y_values[i:i + seq_length]
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
    return np.array(x_sequences), np.array(y_sequences)


# Load dataset
dataset_path = 'datasets/dataset_500_samples_sin.csv'
dataset = pd.read_csv(dataset_path)

# Extract x and y values
x = dataset['x'].values
y = dataset['y'].values

# # Normalize y values
# scaler = MinMaxScaler(feature_range=(-1, 1))
# y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Define sequence length
seq_length = 10

# Create input sequences
x_sequences, y_sequences = create_sequences(x, y, seq_length)

# Split the sequences into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x_sequences, y_sequences, test_size=0.2, random_state=42)
print(x_sequences)
print(y_sequences)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print(x_train_tensor.size())
print(x_test_tensor.size())
