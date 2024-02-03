import csv

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csvPlot
import plot


def save_data_to_csv(predictions, actual_values, days, accuracy, stock, constants, save_path):
    with open(save_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['Stock', 'Day', 'Predicted Price', 'Actual Price', 'Trend Accuracy', 'Model Name', 'Architecture', 'Qubits', 'Layers', 'Seed', 'Lookback', 'Batch Size'])
        for i in range(len(predictions)):
            csv_writer.writerow([f'{stock}', days[i], predictions[i], actual_values[i], accuracy] + constants)


def save_data_to_csv_no_accuracy(predictions, actual_values, days, stock, constants, save_path):
    with open(save_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['Stock', 'Day', 'Predicted Price', 'Actual Price', 'Model Name', 'Architecture', 'Qubits', 'Layers', 'Seed', 'Lookback', 'Batch Size'])
        for i in range(len(predictions)):
            csv_writer.writerow([f'{stock}', days[i], predictions[i], actual_values[i]] + constants)


def save_baseline_points(baseline_points, days, stock, save_path):
    with open(save_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['Stock', 'Day', 'Baseline Point'])
        for i in range(len(baseline_points)):
            csv_writer.writerow([f'{stock}', days[i], baseline_points[i]])


def calculate_accuracy_score(percentage_changes, actual_points, predicted_points, last_actual_value):

    if len(percentage_changes) != len(predicted_points):
        raise ValueError("Input arrays must have the same length")

    correct_predictions = 0
    total_predictions = len(predicted_points)

    actual_trends = np.sign(percentage_changes).values

    changes = [predicted_points[0] - last_actual_value]
    for i in range(1, len(predicted_points)):
        change = predicted_points[i] - actual_points[i - 1]
        changes.append(change)

    predicted_trends = np.sign(changes)

    for i in range(len(predicted_points)):
        if predicted_trends[i] == actual_trends[i]:
            correct_predictions += 1

    return correct_predictions / total_predictions


def calculate_test_mse_loss(actual_points, predicted_points):
    if len(actual_points) != len(predicted_points):
        raise ValueError("Predictions and Actual values must have the same length")

    loss = mean_squared_error(actual_points, predicted_points)
    return loss


def get_predicted_points_from_csv(data_path):
    data = pd.read_csv(data_path)

    # Extract the values from the "Predicted Price" column
    predicted_points = data["Predicted Price"].values.tolist()

    return predicted_points


def get_percentage_change_from_csv(data_path, last_actual_value):
    data = pd.read_csv(data_path)

    predicted_points = data["Predicted Price"].values.tolist()
    actual_points = data["Actual Price"].values.tolist()
    percentage_changes = [(predicted_points[0] / last_actual_value) - 1]

    for i in range(1, len(predicted_points)):
        percentage_changes.append((predicted_points[i] / actual_points[i-1]) - 1)

    return percentage_changes


def get_mean_of_5_seeds(seed1, seed2, seed3, seed4, seed5):
    if len(seed1) != len(seed2) != len(seed3) != len(seed4) != len(seed5):
        raise ValueError("All arrays must have the same length")

    mean_points = []
    for i in range(len(seed1)):
        mean_points.append((seed1[i] + seed2[i] + seed3[i] + seed4[i] + seed5[i]) / 5)

    return mean_points


def get_stock_loss_values(data_path, stock):
    data = pd.read_csv(data_path)

    # Initialize an empty array to store loss values
    loss_values = []

    # Iterate through each row
    for index, row in data.iterrows():
        # Check if the value in the "stock" column matches the specified stock
        if row['stock'] == stock:
            # If it matches, append the corresponding "avg_loss" value to the array
            loss_values.append(row['avg_loss'])

    return loss_values


def get_accumulated_loss_values(data_path):
    data = pd.read_csv(data_path)

    # Group average loss values by epoch and calculate the mean
    grouped_data = data.groupby('epoch')['avg_loss'].mean()

    # Convert the grouped data to an array of loss values
    loss_values = grouped_data.values.tolist()

    return loss_values



