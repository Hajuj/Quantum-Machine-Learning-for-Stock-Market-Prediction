import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error


def evaluate_results(actual, predicted, stock, model):
    plots = f'../plots/evaluation/{model}'
    if not os.path.exists(plots):
        os.makedirs(plots)

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    me = max_error(actual, predicted)

    metrics_names = ['MSE', 'MAE', 'ME']
    metrics_values = [mse, mae, me]

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

    plt.savefig(plots + f'/{stock}.png', dpi=300, format='png')
    plt.show()
