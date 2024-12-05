import torch
import torch.nn as nn

class NLinear(nn.Module):
    """
    Normalized Linear Model
    """
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        # Define linear layers
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        Forward pass for Normalized Linear model
        Args:
            x: [Batch, Seq_len, Channels]
        Returns:
            x: [Batch, Pred_len, Channels]
        """
        seq_last = x[:, -1:, :].detach()  # Get the last sequence value for normalization
        x = x - seq_last  # Normalize input

        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])  # Apply linear layer per channel
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)  # Shared linear layer

        x = x + seq_last  # Add back the last sequence value
        return x  # [Batch, Pred_len, Channels]
