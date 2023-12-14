import torch
import torch.nn as nn


class LSTM(nn.Module):
    # def __init__(self, input_size, hidden_size, batch_first=True):
    #     super(LSTM, self).__init__()
    #     self.hidden_size = hidden_size
    #     self.batch_first = batch_first
    #
    #     # Define the LSTM layer
    #     self.lstm = nn.LSTM(input_size, hidden_size, batch_first=self.batch_first)
    #
    # def forward(self, x):
    #     if self.batch_first:
    #         batch_size, seq_length, features_size = x.size()
    #     else:
    #         seq_length, batch_size, features_size = x.size()
    #
    #     # Initialize hidden and cell states
    #     h_t = torch.zeros(1, batch_size, self.hidden_size)  # hidden state
    #     c_t = torch.zeros(1, batch_size, self.hidden_size)  # cell state
    #
    #     # Forward propagate the LSTM
    #     out, (h_t, c_t) = self.lstm(x, (h_t, c_t))
    #
    #     return out[:, -1, :]  # returning only the last output for simplicity

    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
