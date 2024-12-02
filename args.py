class Args:
    def __init__(self):
        # General parameters
        self.model_type = 'Informer'  # Set to 'Informer' or 'RNN'
        self.data_path = '/Users/scobi/Desktop/SCLoadForecasting/combinedDaytonData_fill.csv'
        self.batch_size = 16
        self.learning_rate = 0.001
        self.epochs = 10
        self.num_workers = 0

        # RNN parameters
        self.rnn_input_size = 4
        self.rnn_hidden_size = 16
        self.rnn_output_size = 1
        self.rnn_num_layers = 1

        # Informer parameters
        self.enc_in = 4  # Input features (temperature, precipitation, humidity, wind_speed)
        self.dec_in = 4
        self.c_out = 1  # Output feature (dayton_mw)
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.d_model = 512  # Model dimensions
        self.n_heads = 8  # Number of attention heads
        self.e_layers = 2  # Number of encoder layers
        self.d_layers = 1  # Number of decoder layers
        self.d_ff = 2048  # Feed-forward network size
        self.factor = 5  # ProbSparse factor
        self.dropout = 0.1  # Dropout rate
        self.embed = 'timeF'  # Embedding type
        self.embed_type = 0  # Embedding variant
        self.freq = 'h'  # Frequency ('h' for hourly)
        self.activation = 'gelu'  # Activation function
        self.output_attention = False
        self.distil = True  # Whether to use distillation
