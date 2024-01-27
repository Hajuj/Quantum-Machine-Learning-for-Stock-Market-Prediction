import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_results(actual, predicted, stock, model):
    plots = f'../plots/evaluation/{model}'
    if not os.path.exists(plots):
        os.makedirs(plots)

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)

    metrics_names = ['MSE', 'MAE', 'RMSE']
    metrics_values = [mse, mae, rmse]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics_names, metrics_values, width=0.3, color=['blue', 'green', 'orange'])
    plt.title(f'Evaluation of stock {stock} ({model})')
    plt.ylabel('Value')

    # Adding metric values on top of each bar with more space
    for bar, metric_value, metric_name in zip(bars, metrics_values, metrics_names):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, metric_name, ha='center', color='black',
                 fontsize=10)

        # Set ylim to avoid bars extending beyond the plot
    plt.ylim(bottom=-0.1, top=max(metrics_values) * 1.2)

    # Remove x-axis values
    plt.xticks([])

    plt.savefig(plots + f'/{stock}/test_loss.png', dpi=300, format='png')
    plt.show()


def show_loss_curve(epochs, loss_values, stock, model):
    plots = f'../plots/evaluation/{model}/{stock}'
    if not os.path.exists(plots):
        os.makedirs(plots)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, marker='o', color='red', label='Loss Curve')
    plt.title(f'Loss Curve for stock {stock} ({model})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.savefig(plots + f'/loss_curve_training.png', dpi=300, format='png')
    plt.show()


def plot_1_day_predictions(predictions, actual_values, time_values, stock, save_path):

    plt.plot(time_values,
             actual_values,
             color='blue',
             label='Actual')

    # Plot the predicted points for the test data
    plt.plot(time_values,
             predictions,
             color='red',
             label='Predicted')


    # Set the locator and formatter for the x-axis
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(f'{stock} Stock Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig(save_path + '/1day.png', dpi=300, format='png', bbox_inches='tight')

    plt.show()


def plot_10_day_prediction(predictions, time_values, actual_values, stock, save_path):

    plt.plot(time_values, actual_values, color='blue', label='Actual')
    plt.plot(time_values, predictions, color='red', label='Predicted')

    # Set the locator and formatter for the x-axis
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(f'{stock} Stock Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig(save_path + '/10day.png', dpi=300, format='png', bbox_inches='tight')

    plt.show()


def plot_loss_curve(data_path, save_path, stock, seed):
    data = pd.read_csv(data_path)

    loss_array = []
    epochs = []

    for index, row in data.iterrows():
        # Check if the value in the "stock" column matches the provided stock parameter
        if row['stock'] == stock:
            # Append the value from the "avg_loss" column to the loss array
            loss_array.append(row['avg_loss'])
            epochs.append(row['epoch'])

    plt.plot(epochs, loss_array)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epochs for Stock: {stock}')

    plt.savefig(save_path + f'/train_loss_seed_{seed}', dpi=300, format='png', bbox_inches='tight')
    plt.show()


def plot_accumulated_loss_curve(data_path, save_path, seed):
    data = pd.read_csv(data_path)

    avg_losses = data.groupby('epoch')['avg_loss'].mean().tolist()
    epochs = sorted(data['epoch'].unique())

    plt.plot(epochs, avg_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.savefig(save_path + f'/accumulated_train_loss_seed_{seed}', dpi=300, format='png', bbox_inches='tight')
    plt.show()


def plot_baseline(predictions, actual_values, time_values, stock, save_path):
    plt.plot(time_values,
             actual_values,
             color='blue',
             label='Actual')

    # Plot the predicted points for the test data
    plt.plot(time_values[10:],
             predictions[:-10],
             color='green',
             label='Baseline')

    # Set the locator and formatter for the x-axis
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.title(f'{stock} Stock Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig(save_path + '/baseline.png', dpi=300, format='png', bbox_inches='tight')

    plt.show()
