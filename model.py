import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


from LTSF.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from LTSF.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from LTSF.layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp

def create_time_series_dataset(data, lookback_window, forecasting_horizon=1, test_size=0.25):
    x_processed = []
    y_processed = []

    # Create sliding windows
    for i in range(len(data) - lookback_window - forecasting_horizon + 1):
        x_window = data[i:i + lookback_window, :-1]  # Use all features except target
        y_value = data[i + lookback_window + forecasting_horizon - 1, -1]  # Target is the load forecast
        x_processed.append(x_window)
        y_processed.append(y_value)

    # Convert to numpy arrays
    x_processed = np.array(x_processed)
    y_processed = np.array(y_processed)

    # Print shapes and contents of x_processed and y_processed
    print("Shape of x_processed (input windows):", x_processed.shape)  # Expected: (num_samples, lookback_window, num_features)
    print("Sample x_processed (first input window):", x_processed[0])  # Prints first input window
    print("Shape of y_processed (targets):", y_processed.shape)        # Expected: (num_samples,)
    print("Sample y_processed (first target):", y_processed[0])        # Prints first target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_processed, y_processed, test_size=test_size, shuffle=False)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False)

    return train_loader, test_loader

class SimpleRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=1, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #rnn layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        #output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class InformerModel(nn.Module):
    def __init__(self, configs):
        super(InformerModel, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
