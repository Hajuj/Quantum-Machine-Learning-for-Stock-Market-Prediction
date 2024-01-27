from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error


def get_baseline_points(test_loader, scaler):
    reg = LinearRegression()
    baseline_points = []

    for X_batch, y_batch in test_loader:
        X_batch_flat = X_batch.view(X_batch.size(0), -1)
        y_batch = y_batch.reshape(-1, 1)

        reg.fit(X_batch_flat, y_batch)
        baseline_points.extend(reg.predict(X_batch_flat))

    baseline_array = np.array(baseline_points)
    dummy_array = np.zeros((len(baseline_array), scaler.n_features_in_))
    dummy_array[:, 0] = baseline_array.ravel()
    denormalized_predictions = scaler.inverse_transform(dummy_array)[:, 0].flatten()

    return denormalized_predictions

