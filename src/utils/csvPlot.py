import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


# Price Plots
def plot_all_models_price_1day(stock, days, actual_points, qlstm_points, lstm_points, qrnn_points, baseline_points, save_path):
    plt.figure(figsize=(10, 6))

    # Plot actual points
    plt.plot(days, actual_points, label='Actual Price', color='blue')

    # Plot points for different models
    plt.plot(days, qlstm_points, label='QLSTM', color='red')
    plt.plot(days, lstm_points, label='LSTM', color='green')
    plt.plot(days, qrnn_points, label='QRNN', color='purple')
    plt.plot(days, baseline_points, label='Baseline', color='black')

    # Add labels and title
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(f'1 day Price Prediction Comparison on stock {stock}')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    # Add legend
    plt.legend()
    plt.savefig(save_path + f'/{stock}1day.png', dpi=300, format='png', bbox_inches='tight')


def plot_all_models_price_10day(stock, days, actual_points, qlstm_points, lstm_points, qrnn_points, baseline_points, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(days, actual_points, label='Actual Price', color='blue')
    plt.plot(days, qlstm_points, label='QLSTM', color='red')
    plt.plot(days, baseline_points, label='Baseline', color='black')
    plt.plot(days, lstm_points, label='LSTM', color='green')
    plt.plot(days, qrnn_points, label='QRNN', color='purple')

    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(f'10 day Price Prediction Comparison on stock {stock}')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(save_path + f'/{stock}10day.png', dpi=300, format='png', bbox_inches='tight')


def plot_percentage_change_1day(stock, days, actual_changes, qlstm_changes, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(days, actual_changes, label='Actual', color='blue')
    plt.plot(days, qlstm_changes, label='QLSTM', color='red')

    plt.xlabel('Days')
    plt.ylabel('Percentage Change')
    plt.title(f'Percentage Change comparison on stock {stock}')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(save_path + f'/{stock}PercentReturn.png', dpi=300, format='png', bbox_inches='tight')


def plot_predictions_with_mean_curve(stock, days, model_data, actual_points, save_path):
    # Define colors for the plot
    model_colors = ['red', 'green', 'black', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    # Plot actual points
    plt.figure(figsize=(10, 6))
    plt.plot(days, actual_points, label='Actual', color='blue')

    # Iterate over each architecture and its data
    for i, (architecture, data) in enumerate(model_data.items()):
        # Extract mean, min, and max predictions
        mean_predictions = data['mean']
        min_predictions = data['min']
        max_predictions = data['max']

        # Define color for the model
        model_color = model_colors[i % len(model_colors)]

        # Plot the area between min and max predictions
        plt.fill_between(days, min_predictions, max_predictions, alpha=0.3, color=model_color)

        # Plot the mean predictions
        plt.plot(days, mean_predictions, label=f'{architecture} Mean', linewidth=2,
                 color=model_color)

    # Add labels and legend
    plt.xlabel('Time Step')
    plt.ylabel('Predicted Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.title(f'Model Predictions with Mean Curves for stock {stock}')
    plt.legend()

    # Save plot
    plt.savefig(save_path + f'/{stock}MeanCurve.png', dpi=300, format='png', bbox_inches='tight')


# Performance Plots

def plot_all_models_performance(qlstm_loss, lstm_loss, qrnn_loss, baseline_loss, save_path):
    models = ['QLSTM', 'LSTM', 'QRNN', 'Baseline']

    # Loss values for each model
    loss_values = [qlstm_loss, lstm_loss, qrnn_loss, baseline_loss]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(models, loss_values, color=['green', 'brown', 'red', 'black'])
    plt.xlabel('Models')
    plt.ylabel('Loss Values')
    plt.title('Performance of Different Models')
    plt.legend()
    plt.savefig(save_path + f'/AllModelsPerformance.png', dpi=300, format='png', bbox_inches='tight')


def plot_qlstm_performance(selected_models_with_loss_value, save_path):
    # Extract models and loss values from the input array
    models = [entry[0] for entry in selected_models_with_loss_value]
    loss_values = [entry[1] for entry in selected_models_with_loss_value]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, loss_values)

    # Assign different colors to each bar
    colors = ['green', 'blue', 'red', 'black', 'brown', 'orange', 'purple', 'pink']
    for bar, color in zip(bars, colors[:len(models)]):
        bar.set_color(color)

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('Loss Values')
    plt.title('Performance of different QLSTM Architectures')

    # Create legend
    plt.legend(bars, models)

    # Save plot
    plt.savefig(save_path + '/QLSTMArchsPerformance.png', dpi=300, format='png', bbox_inches='tight')


def plot_heatmap(selected_stocks_with_result_file, save_path):
    # Create a dictionary to hold the mean accuracy values for each model
    accuracy_dict = {}

    # Iterate through each entry in selected_stocks_with_result_file
    for stock, test_file_path in selected_stocks_with_result_file:
        # Extract model information from the file name
        file_name = test_file_path.split("/")[-1]  # Extract file name from the file path
        arch = file_name.split("_")[1]  # Extract the 'arch' parameter from the file name
        lookback = file_name.split("_")[2].split(".")[0]  # Extract the 'lookback' parameter from the file name

        # Combine model information
        model = f"{arch} {lookback}"

        # Load the accuracy data from the CSV file and extract the first accuracy value
        accuracy_data = pd.read_csv(test_file_path)['Trend Accuracy'][0]

        # Check if the stock already exists in the dictionary
        if stock not in accuracy_dict:
            accuracy_dict[stock] = {}

        # Check if the model already exists for this stock
        if model not in accuracy_dict[stock]:
            accuracy_dict[stock][model] = []

        # Append the accuracy value to the corresponding list for the model within the stock
        accuracy_dict[stock][model] = accuracy_data

    # Convert the dictionary to a DataFrame
    accuracy_df = pd.DataFrame.from_dict(accuracy_dict, orient='index')

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mean Accuracy Heatmap')
    plt.xlabel('Model')
    plt.ylabel('Stocks')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent overlap of labels
    plt.savefig(save_path + '/Heatmap1.png', dpi=300, format='png', bbox_inches='tight')


# Loss Plots

def plot_mean_loss_curve(epochs, loss_values, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, loss_values, label='', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Loss')
    plt.title(f'Mean Loss of all stocks')
    plt.legend()
    plt.savefig(save_path + '/MeanLoss.png', dpi=300, format='png', bbox_inches='tight')


def plot_stock_loss_curve(epochs, loss_values, stock, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, loss_values, label='', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss over stock {stock}')
    plt.legend()
    plt.savefig(save_path + f'/{stock}LossCurve.png', dpi=300, format='png', bbox_inches='tight')