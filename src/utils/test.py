import numpy as np
import torch

import preprocess


def test_model_10day(model, last_sequence, scaler, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    with torch.no_grad():
        output = model(last_sequence)
        new_sequence = last_sequence
        predictions.append(output)
        for _ in range(9):
            new_sequence = preprocess.create_new_sequence(new_sequence, output)
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
