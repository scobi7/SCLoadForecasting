import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on both ends
        front = x[:, :, :1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=2)
        x = self.avg(x)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        residual = x - trend
        return residual, trend


class DLinear(nn.Module):
    """
    Decomposition-Linear (DLinear) Model
    """
    def __init__(self, args):
        super(DLinear, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.kernel_size = args.moving_avg_kernel  # Add this to args
        self.decomp = SeriesDecomp(self.kernel_size)
        self.individual = args.individual

        if self.individual:
            # Separate Linear layers for each feature
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            # Shared Linear layers across all features
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [batch_size, seq_len, channels]
        batch_size, seq_len, channels = x.shape

        # Decompose into seasonal and trend components
        residual, trend = self.decomp(x.permute(0, 2, 1))  # [batch_size, channels, seq_len]
        
        residual = residual.permute(0, 2, 1)  # Back to [batch_size, seq_len, channels]
        trend = trend.permute(0, 2, 1)  # Back to [batch_size, seq_len, channels]

        if self.individual:
            # Individual Linear layers per feature
            seasonal_output = torch.cat(
                [self.linear_seasonal[i](residual[:, :, i]).unsqueeze(-1) for i in range(channels)], dim=-1
            )
            trend_output = torch.cat(
                [self.linear_trend[i](trend[:, :, i]).unsqueeze(-1) for i in range(channels)], dim=-1
            )
        else:
            # Shared Linear layers
            residual = residual.reshape(-1, seq_len)  # Flatten to [batch_size * channels, seq_len]
            trend = trend.reshape(-1, seq_len)  # Flatten to [batch_size * channels, seq_len]
            
            seasonal_output = self.linear_seasonal(residual)  # [batch_size * channels, pred_len]
            trend_output = self.linear_trend(trend)  # [batch_size * channels, pred_len]
            
            seasonal_output = seasonal_output.reshape(batch_size, self.pred_len, channels)
            trend_output = trend_output.reshape(batch_size, self.pred_len, channels)

        # Combine seasonal and trend components
        output = seasonal_output + trend_output
        return output
