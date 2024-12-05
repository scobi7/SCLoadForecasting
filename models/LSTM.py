import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM Model for Time Series Forecasting
    """
    def __init__(self, configs):
        super(LSTMModel, self).__init__()
        self.input_size = configs.enc_in  # Number of input features
        self.hidden_size = configs.hidden_size  # Hidden layer size
        self.num_layers = configs.num_layers  # Number of LSTM layers
        self.pred_len = configs.pred_len  # Prediction length
        self.seq_len = configs.seq_len  # Sequence length
        self.batch_size = configs.batch_size
        self.output_size = configs.dec_in  # Output size (forecast channels)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully connected layer to map hidden state to predictions
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Forward pass for LSTM
        Args:
            x: [Batch, Seq_len, Features]
        Returns:
            [Batch, Pred_len, Features]
        """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM output
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # lstm_out: [Batch, Seq_len, Hidden_size]

        # Select the last output of the LSTM for each sequence
        last_hidden = lstm_out[:, -1, :]  # [Batch, Hidden_size]

        # Map to prediction
        output = self.fc(last_hidden)  # [Batch, Output_size]

        # Repeat predictions for pred_len timesteps
        output = output.unsqueeze(1).repeat(1, self.pred_len, 1)  # [Batch, Pred_len, Output_size]

        return output
