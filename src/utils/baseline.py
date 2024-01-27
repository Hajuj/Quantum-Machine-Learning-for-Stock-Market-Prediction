from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error


def get_baseline_points(data):
    closing_price = data['Close']

    date_ordinal = data['Time'].apply(lambda x: x.toordinal())

    date_ordinal_reshaped = np.array(date_ordinal).reshape(-1, 1)
    closing_price_reshaped = np.array(closing_price)
    reg = LinearRegression()
    reg.fit(date_ordinal_reshaped, closing_price_reshaped)
    baseline_points = reg.predict(date_ordinal_reshaped)

    # Mean Squared Error for the Baseline(Linear Regression)
    mse = mean_squared_error(closing_price_reshaped, baseline_points)
    print(f'MSE - Baseline: {mse}')

    return baseline_points

