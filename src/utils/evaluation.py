import csv

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates
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


# Calculations

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

# Read data from csv


def get_predicted_points_from_csv(data_path):
    data = pd.read_csv(data_path)

    # Extract the values from the "Predicted Price" column
    predicted_points = data["Predicted Price"].values.tolist()

    return predicted_points


def get_actual_points_from_csv(data_path):
    data = pd.read_csv(data_path)

    actual_points = data["Actual Price"].values.tolist()

    return actual_points


def get_days_from_csv(data_path):
    data = pd.read_csv(data_path)

    days = data["Day"].values.tolist()
    return days


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


def generate_area_plot_data(predictions_dictionary):
    # Initialize dictionaries to store mean, min, and max values for each architecture
    architecture_data = {}

    # Iterate over each architecture and its predictions
    for architecture, seeds_predictions in predictions_dictionary.items():
        # Concatenate predictions from all seeds along the first axis
        all_predictions = np.array(seeds_predictions)

        # Calculate mean, min, and max values across all seeds
        mean_predictions = np.mean(all_predictions, axis=0)
        min_predictions = np.min(all_predictions, axis=0)
        max_predictions = np.max(all_predictions, axis=0)

        # Store mean, min, and max values for the architecture
        architecture_data[architecture] = {
            'mean': mean_predictions,
            'min': min_predictions,
            'max': max_predictions
        }

    return architecture_data


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


ex_stock = 'NVDA'
stocks = ['NVDA', 'DIS', 'KO', 'MO', 'BABA', 'MA', 'V', 'JPM', 'PG', 'TSM', 'META', 'TSLA', 'MSFT', 'AAPL', 'ABBV',
          'PEP', 'CRM', 'PFE', 'NFLX', 'AMD', 'ABT', 'PM', 'BA', 'NKE', 'GS', 'T', 'C', 'MU']
data_path = "../results/test/QLSTM/NVDA/1_arch1.2_lookback5_seed5.csv"
actual_points = get_actual_points_from_csv(data_path)
days = get_days_from_csv(data_path)
days_num = mdates.date2num(days)

# Mean Curve Prediction


arch1_2_lookback5_points_seed1 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback5_seed1.csv')
arch1_2_lookback5_points_seed2 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback5_seed2.csv')
arch1_2_lookback5_points_seed3 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback5_seed3.csv')
arch1_2_lookback5_points_seed4 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback5_seed4.csv')
arch1_2_lookback5_points_seed5 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback5_seed5.csv')

arch1_3_lookback5_points_seed1 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback5_seed1.csv')
arch1_3_lookback5_points_seed2 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback5_seed2.csv')
#arch1_3_lookback5_points_seed3 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback5_seed3.csv')
#arch1_3_lookback5_points_seed4 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback5_seed4.csv')
#arch1_3_lookback5_points_seed5 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback5_seed5.csv')

arch1_2_lookback10_points_seed1 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback10_seed1.csv')
arch1_2_lookback10_points_seed2 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback10_seed2.csv')
arch1_2_lookback10_points_seed3 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback10_seed3.csv')
arch1_2_lookback10_points_seed4 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback10_seed4.csv')
arch1_2_lookback10_points_seed5 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.2_lookback10_seed5.csv')

arch1_3_lookback10_points_seed1 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback10_seed1.csv')
arch1_3_lookback10_points_seed2 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback10_seed2.csv')
arch1_3_lookback10_points_seed3 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback10_seed3.csv')
arch1_3_lookback10_points_seed4 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback10_seed4.csv')
arch1_3_lookback10_points_seed5 = get_predicted_points_from_csv(f'../results/test/QLSTM/NVDA/1_arch1.3_lookback10_seed5.csv')


predictions_dict = {
    'Arch1_2_lookback5': [arch1_2_lookback5_points_seed1, arch1_2_lookback5_points_seed2, arch1_2_lookback5_points_seed3, arch1_2_lookback5_points_seed4, arch1_2_lookback5_points_seed5],
    'Arch1_2_lookback10': [arch1_2_lookback10_points_seed1, arch1_2_lookback10_points_seed2, arch1_2_lookback10_points_seed3, arch1_2_lookback10_points_seed4, arch1_2_lookback10_points_seed5],
    'Arch1_3_lookback5': [arch1_3_lookback5_points_seed1, arch1_3_lookback5_points_seed2],
    'Arch1_3_lookback10': [arch1_3_lookback10_points_seed1, arch1_3_lookback10_points_seed2, arch1_3_lookback10_points_seed3, arch1_3_lookback10_points_seed4, arch1_3_lookback10_points_seed5]
}

# model_data = generate_area_plot_data(predictions_dict)
save_path = 'C:/Users/Leonard Niessen/Projects/src/plots'
model_data = generate_area_plot_data(predictions_dict)
csvPlot.plot_predictions_with_mean_curve(ex_stock, days_num, model_data, actual_points, save_path)

# Heatmap
selected_stocks = ['AAPL', 'KO', 'BABA', 'MA', 'PG', 'PFE', 'NKE', 'TSLA', 'T', 'PM']
according_test_files = []
stocks_with_test_file = []
# csvPlot.plot_heatmap()




