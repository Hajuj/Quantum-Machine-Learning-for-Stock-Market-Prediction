import numpy as np
import matplotlib.pyplot as plt
import os
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


