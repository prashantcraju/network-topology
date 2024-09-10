class LSTM(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, output_size=10, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initial hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initial cell state
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Instantiate the network
lstm_model = LSTM()
print(lstm_model)
