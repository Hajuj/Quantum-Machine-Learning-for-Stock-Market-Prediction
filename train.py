# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
# from copy import deepcopy as dc
#
# import preprocess
# from qlstm import QLSTM
#
# #
# # def train_qlstm(model, train_loader, test_loader, epochs, learning_rate):
# #     criterion = nn.MSELoss()
# #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# #
# #     for epoch in range(epochs):
# #         model.train()
# #         total_loss = 0
# #
# #         for x_batch, y_batch in train_loader:
# #             optimizer.zero_grad()
# #             y_pred_tuple = model(x_batch)
# #             y_pred = y_pred_tuple[0]  # Assuming hidden_seq is the first element of the tuple
# #
# #             # Modify this line based on your specific requirement
# #             y_pred_last = y_pred[:, -1, :]  # Taking the last value in each sequence
# #
# #             loss = criterion(y_pred_last, y_batch)
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
# #
# #         if (epoch + 1) % 2 == 0:
# #             model.eval()
# #             total_test_loss = 0
# #             with torch.no_grad():
# #                 for x_test, y_test in test_loader:
# #                     test_pred_tuple = model(x_test)
# #                     test_pred = test_pred_tuple[0]  # Assuming hidden_seq is the first element of the tuple
# #
# #                     # Modify this line based on your specific requirement
# #                     test_pred_last = test_pred[:, -1, :]  # Taking the last value in each sequence
# #
# #                     test_loss = criterion(test_pred_last, y_test)
# #                     total_test_loss += test_loss.item()
# #
# #             avg_loss = total_loss / len(train_loader)
# #             avg_test_loss = total_test_loss / len(test_loader)
# #             print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
# #
# #     predictions, actuals = [], []
# #     model.eval()
# #     with torch.no_grad():
# #         for x_test, y_test in test_loader:
# #             test_pred_tuple = model(x_test)
# #             test_pred = test_pred_tuple[0]  # Assuming hidden_seq is the first element of the tuple
# #             test_pred_last = test_pred[:, -1, :]  # Taking the last value in each sequence
# #
# #             # Store predictions and actual values
# #             predictions.extend(test_pred_last.numpy())
# #             actuals.extend(y_test.numpy())
# #
# #     return predictions, actuals
# #
# #
# # # Example usage
# # qlstm_model = QLSTM(input_size=1, hidden_size=10)
# # train_qlstm(qlstm_model, preprocess.train_loader, preprocess.test_loader, 20, 0.001)
# #
# #
# # # Train the model and get predictions and actual values
# # predictions, actuals = train_qlstm(qlstm_model, preprocess.train_loader, preprocess.test_loader, 20, 0.001)
# #
# # # Convert predictions and actuals to a suitable format for plotting
# # predictions = np.array(predictions).flatten()
# # actuals = np.array(actuals).flatten()
# #
# # # Plotting
# # plt.figure(figsize=(10, 6))
# # plt.plot(predictions, label='Predictions', color='blue')
# # plt.plot(actuals, label='Actual Values', color='orange')
# # plt.title('Predictions vs Actual Values')
# # plt.xlabel('Sample')
# # plt.ylabel('Value')
# # plt.legend()
# # plt.show()
#
# # Set device to GPU if available, else CPU
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Initialize the QLSTM model with specified input and hidden layer sizes
# model = QLSTM(input_size=1, hidden_size=10)
# model.to(device)
#
#
# def train_one_epoch():
#     """
#     Train the model for one epoch.
#     """
#     model.train(True)
#     print(f'Epoch: {epoch + 1}')
#     running_loss = 0.0
#     total_batches = len(preprocess.train_loader)
#
#     for batch_index, batch in enumerate(preprocess.train_loader):
#         x_batch, y_batch = batch[0].to(device), batch[1].to(device)
#
#         # Forward pass - Only use the hidden_seq part of the output
#         hidden_seq, _ = model(x_batch)  # hidden_seq is the output sequence
#
#         # Select the last output for each sequence in the batch for loss calculation
#         last_outputs = hidden_seq[:, -1, :].squeeze()  # shape: [batch_size, hidden_size]
#
#         # Compute loss
#         loss = loss_function(last_outputs, y_batch)
#         running_loss += loss.item()
#
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Print loss every 5 batches
#         if batch_index % 5 == 4:
#             avg_loss_across_batches = running_loss / 5
#             print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1,
#                                                     avg_loss_across_batches))
#             running_loss = 0.0
#     print()
#
#     avg_epoch_loss = running_loss / total_batches
#     return avg_epoch_loss
#
#
# def test_one_epoch():
#     """
#     Test the model for one epoch.
#     """
#
#     model.eval()  # Set the model to evaluation mode
#     running_loss = 0.0
#     total_batches = len(preprocess.test_loader)
#
#     # Loop through each batch in the test data
#     for batch_index, batch in enumerate(preprocess.test_loader):
#         x_batch, y_batch = batch[0].to(device), batch[1].to(device)
#
#         with torch.no_grad():
#             hidden_seq, _ = model(x_batch)  # hidden_seq is the output sequence
#
#             # Select the last output for each sequence in the batch for loss calculation
#             last_outputs = hidden_seq[:, -1, :].squeeze()  # shape: [batch_size, hidden_size]
#
#             # Compute loss
#             loss = loss_function(last_outputs, y_batch)
#             running_loss += loss.item()
#
#     avg_loss_across_batches = running_loss / len(preprocess.test_loader)
#     print('Test Loss: {0:.3f}'.format(avg_loss_across_batches))
#     print('***************************************************')
#     print()
#
#     avg_epoch_loss = running_loss / total_batches
#     return avg_epoch_loss
#
#
# # Training parameters
# learning_rate = 0.001
# num_epochs = 10
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # Main training loop
# train_losses = []
# test_losses = []
#
# for epoch in range(num_epochs):
#     train_loss = train_one_epoch()
#     val_loss = test_one_epoch()
#
#     train_losses.append(train_loss)
#     test_losses.append(val_loss)
#
#     print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.3f}, Test Loss: {val_loss:.3f}")
#
# # # Plot training and test loss
# # plt.plot(train_losses, label='Training Loss')
# # plt.plot(test_losses, label='Test Loss')
# # plt.title('Training and Test Loss per Epoch')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.show()
#
#
# # Function to perform predictions on the dataset
# def predict(dataset_loader):
#     model.eval()  # Set the model to evaluation mode
#     predictions = []
#     actual = []
#
#     with torch.no_grad():
#         for batch in dataset_loader:
#             x_batch, y_batch = batch[0].to(device), batch[1]
#
#             hidden_seq, _ = model(x_batch)
#             last_outputs = hidden_seq[:, -1, :].squeeze()
#
#             # Store CPU data for plotting
#             predictions.extend(last_outputs.to('cpu').numpy())
#             actual.extend(y_batch.numpy())
#
#     return actual, predictions
#
#
# # Run predictions on the test set
# actual, predicted = predict(preprocess.test_loader)
#
# # Plot actual vs predicted values
# plt.figure(figsize=(10, 5))
# plt.plot(actual, label='Actual Data', marker='.')
# plt.plot(predicted, label='Predicted Data', marker='x')
# plt.title('Actual vs Predicted Data')
# plt.xlabel('Data Point Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
#
# # # Generate predictions for training set
# # with torch.no_grad():
# #     # Get the model's output, which is a tuple
# #     hidden_seq, _ = model(preprocess.X_train.to(device))
# #
# #     # Select the last output for each sequence in the batch
# #     predicted = hidden_seq[:, -1, :].squeeze()  # shape: [batch_size, hidden_size]
# #
# #     # Move the tensor to CPU and convert to numpy
# #     predicted = predicted.to('cpu').numpy()
# #
# # # Plot actual vs predicted for training set
# # plt.plot(preprocess.y_train, label='Actual')
# # plt.plot(predicted, label='Predicted')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.legend()
# # plt.show()
# #
# # # Additional processing for visualization
# # train_predictions = predicted.flatten()
# #
# # dummies = np.zeros((preprocess.X_train.shape[0], preprocess.sequence_length + 1))
# # dummies[:, 0] = train_predictions
# # scaler = MinMaxScaler(feature_range=(-1, 1))
# # dummies = scaler.inverse_transform(dummies)
# #
# # train_predictions = dc(dummies[:, 0])
# #
# # dummies = np.zeros((preprocess.X_train.shape[0], preprocess.sequence_length + 1))
# # dummies[:, 0] = preprocess.y_train.flatten()
# # dummies = scaler.inverse_transform(dummies)
# #
# # new_y_train = dc(dummies[:, 0])
# #
# # plt.plot(new_y_train, label='Actual')
# # plt.plot(train_predictions, label='Predicted')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.legend()
# # plt.show()
# #
# # test_predictions = model(preprocess.X_test.to(device)).detach().cpu().numpy().flatten()
# #
# # dummies = np.zeros((preprocess.X_test.shape[0], preprocess.sequence_length + 1))
# # dummies[:, 0] = test_predictions
# # dummies = scaler.inverse_transform(dummies)
# #
# # test_predictions = dc(dummies[:, 0])
# #
# # dummies = np.zeros((preprocess.X_test.shape[0], preprocess.sequence_length + 1))
# # dummies[:, 0] = preprocess.y_test.flatten()
# # dummies = scaler.inverse_transform(dummies)
# #
# # new_y_test = dc(dummies[:, 0])
# #
# # plt.plot(new_y_test, label='Actual')
# # plt.plot(test_predictions, label='Predicted')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.legend()
# # plt.show()
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from qlstm import QLSTM
import preprocess

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# Initialize the QLSTM model
input_size = 1
hidden_size = 1
n_qubits = 4
n_qlayers = 2
model = QLSTM(input_size, hidden_size, n_qubits=n_qubits, n_qlayers=n_qlayers)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)


def train_model(model, train_loader, loss_function, optimizer, n_epochs):
    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            # print("Data from: 'output': ", output)
            # print("Data from: 'y_batch': ", y_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")


def test_model(model, test_loader, loss_function):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    predictions = []
    with torch.no_grad():  # No need to track gradients during testing
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            test_loss += loss.item()
            predictions.append(output)

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return predictions


# # Check a single batch from the train_loader
# train_features, train_labels = next(iter(preprocess.train_loader))
# print("Train Features Shape:", train_features.shape)
# print("Train Labels Shape:", train_labels.shape)
# print("First few Train Features:", train_features[:5])
# print("First few Train Labels:", train_labels[:5])
#
# # Check a single batch from the test_loader
# test_features, test_labels = next(iter(preprocess.test_loader))
# print("Test Features Shape:", test_features.shape)
# print("Test Labels Shape:", test_labels.shape)
# print("First few Test Features:", test_features[:5])
# print("First few Test Labels:", test_labels[:5])


n_epochs = 20

# Training the model
train_model(model, preprocess.train_loader, loss_function, optimizer, n_epochs)

# Testing the model
predictions = test_model(model, preprocess.test_loader, loss_function)

# Load the entire dataset (x and y values)
data = pd.read_csv(preprocess.data_path)
x_values = data['timesteps'].values
y_values = data['data'].values

# Calculate the starting index for test data
batch_size = preprocess.batch_size
num_train_batches = len(preprocess.train_loader)
train_data_length = batch_size * num_train_batches
# print(train_data_length)

# Flatten the list of predictions for plotting
predicted_points = torch.cat(predictions, dim=0).view(-1)

dummies = np.zeros((preprocess.X_train.shape[0], preprocess.sequence_length + 1))
dummies[:, 0] = predicted_points
dummies = preprocess..inverse_transform(dummies)

# Plot the entire sine curve
plt.plot(x_values, y_values, label='Actual')

# print(len(x_values[train_data_length:train_data_length + len(predicted_points)]))
# print(len(predicted_points.numpy()))

# Plot the predicted points for the test data
plt.scatter(x_values[train_data_length:train_data_length + len(predicted_points)],
            predicted_points.numpy(),
            color='red',
            label='Predicted',
            s=10)

plt.title('Apple Stock Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()
