import torch
import torch.nn as nn

import preprocess
from qlstm import QLSTM


def train_qlstm(model, x_train, y_train, x_test, y_test, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(x_test)
                test_loss = criterion(test_pred, y_test)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')


qlstm_model = QLSTM(input_size=1, hidden_size=10)
train_qlstm(qlstm_model, preprocess.x_train_tensor, preprocess.y_train_tensor, preprocess.x_test_tensor, preprocess.y_test_tensor, 20, 0.001)
