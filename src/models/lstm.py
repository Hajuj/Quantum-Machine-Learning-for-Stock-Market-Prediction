import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    # Set the seed for reproducibility
    def set_seed(self, seed):
        torch.manual_seed(seed)


if __name__ == '__main__':
    lstm = LSTM(2, 1, 1)
    for parameter in lstm.parameters():
        print(parameter)

    model_parameters = filter(lambda p: p.requires_grad, lstm.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
