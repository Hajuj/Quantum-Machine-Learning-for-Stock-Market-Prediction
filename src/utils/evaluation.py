import csv

import numpy as np


def calculate_accuracy_score(percentage_changes, predicted_points, last_actual_value):

    if len(percentage_changes) != len(predicted_points):
        raise ValueError("Input arrays must have the same length")

    correct_predictions = 0
    total_predictions = len(predicted_points)

    actual_trends = np.sign(percentage_changes).values

    changes = [predicted_points[0] - last_actual_value]
    for i in range(1, len(predicted_points)):
        change = predicted_points[i] - predicted_points[i - 1]
        changes.append(change)

    predicted_trends = np.sign(changes)

    for i in range(len(predicted_points)):
        if predicted_trends[i] == actual_trends[i]:
            correct_predictions += 1

    return correct_predictions / total_predictions


def save_data_to_csv(predictions, actual_values, days, accuracy, stock, constants, save_path):
    with open(save_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['Stock', 'Day', 'Predicted Price', 'Actual Price', 'Accuracy', 'Model Name', 'Architecture', 'Layers', 'Seed', 'Lookback', 'Batch Size'])
        for i in range(len(predictions)):
            csv_writer.writerow([f'{stock}', days[i], actual_values[i], predictions[i], accuracy] + constants)


def save_data_to_csv_no_accuracy(predictions, actual_values, days, stock, constants, save_path):
    with open(save_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ['Stock', 'Day', 'Predicted Price', 'Actual Price', 'Model Name', 'Architecture', 'Layers', 'Seed', 'Lookback', 'Batch Size'])
        for i in range(len(predictions)):
            csv_writer.writerow([f'{stock}', days[i], actual_values[i], predictions[i]] + constants)
