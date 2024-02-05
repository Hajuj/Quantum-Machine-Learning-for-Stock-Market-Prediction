import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import preprocess
import preprocess8inputs


def test_model_10day(model, last_sequence, scaler, model_path, sequence_length, arch):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    with torch.no_grad():
        output = model(last_sequence)
        new_sequence = last_sequence
        predictions.append(output)
        for _ in range(9):
            if arch == "1.1" or arch == "1.2" or arch == "1.3" or arch == "1.4" or arch == "2.1" or arch == "2.2" or arch == "2.3" or arch == "2.4":
                new_sequence = preprocess.update_recurrent_sequence(sequence_length, new_sequence, output)
            elif arch == "3.1" or arch == "3.2" or arch == "3.3" or arch == "3.4" or arch == "4.1" or arch == "4.2" or arch == "4.3" or arch == "4.4":
                new_sequence = preprocess8inputs.update_recurrent_sequence(sequence_length, new_sequence, output)
            else:
                print("Invalid architecture during testing 10 day model!!")
                exit()

            output = model(new_sequence)
            predictions.append(output)

    predicted_points = torch.cat(predictions, dim=0).view(-1).numpy()
    dummy_array = np.zeros((len(predicted_points), scaler.n_features_in_))
    dummy_array[:, 0] = predicted_points  # Assuming target variable is the first feature

    # Apply inverse transform to the dummy array
    denormalized_predictions = scaler.inverse_transform(dummy_array)[:, 0].flatten()

    return denormalized_predictions


def test_model(model, test_loader, loss_function, scaler, model_path):
    model.load_state_dict(torch.load(model_path))
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


def test_ibmq_model(model, test_loader, loss_function, scaler, model_path, data_path, train_ratio, arch, seed, sequence_length):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    # Get correct indexes from test data
    data = pd.read_csv(data_path)
    data['Time'] = pd.to_datetime(data['Time'])  # Convert the 'Time' column to datetime objects

    # Convert 'Time' to the format matplotlib requires
    x_values = data['Time'].values
    y_values = data['Close'].values

    # Get the length of the training data
    train_data_length = len(data[:int(len(data) * train_ratio)])

    # Calculate the starting index for test data
    x_test_area = x_values[train_data_length:train_data_length + 153]
    y_test_area = y_values[train_data_length:train_data_length + 153]

    # Create directory
    test_ibmq_file_path = '../results/test/QLSTM_IBMQ/AAPL'
    if not os.path.exists(test_ibmq_file_path):
        os.makedirs(test_ibmq_file_path)

    # Create CSV file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"QLSTM_IBMQ_arch{arch}_seed{seed}_lookback{sequence_length}_{timestamp}.csv"
    test_ibmq_file_path = os.path.join('../results/test/QLSTM_IBMQ/AAPL', file_name)

    # Open the CSV file
    with open(test_ibmq_file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Day', 'Predicted Price', 'Actual Price'])

        with torch.no_grad():  # No need to track gradients during testing
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                loss = loss_function(output, y_batch)
                test_loss += loss.item()

                # Denormalize the output here before saving
                output_denorm = output.numpy()  # Convert to numpy array if it's not already
                dummy_array = np.zeros((len(output_denorm), scaler.n_features_in_))
                dummy_array[:, 0] = output_denorm.flatten()  # Assuming target variable is the first feature
                denormalized_output = scaler.inverse_transform(dummy_array)[:, 0]  # Apply inverse transform

                # Write the denormalized data to the CSV file
                for i in range(len(denormalized_output)):
                    csv_writer.writerow([f'{x_test_area[i]}', f'{denormalized_output[i]}', f'{y_test_area[i]}'])

    return denormalized_output
