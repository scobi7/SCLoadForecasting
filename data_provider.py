import pandas as pd
from torch.utils.data import DataLoader
from informer_data import InformerDataset
from LoadForecasting import LoadForecastingSNN
from sklearn.model_selection import train_test_split

def data_provider(args, flag, limit=None):
    df = pd.read_csv(args.data_path, parse_dates=['datetime'])
    df = df[['datetime', 'temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']]
    
    if limit:  # Limit dataset size for testing
        df = df.head(limit)
    
    train_data, test_data = train_test_split(df, test_size=0.25, shuffle=False)

    if flag == 'train':
        data_to_use = train_data
        shuffle_flag = True
        drop_last = True
    else:  # test
        data_to_use = test_data
        shuffle_flag = False
        drop_last = False

    if args.model_type == 'Informer':
        data_set = InformerDataset(
            data=data_to_use,
            look_back=args.seq_len,
            pred_len=args.pred_len,
            features='M',
            target='dayton_mw',
            freq=args.freq
        )
    else:  # RNN
        data_set = LoadForecastingSNN(data=data_to_use, look_back=5)

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,  # Use batch size from args
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    # Print the batch size and number of batches to verify
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Samples in Dataset: {len(data_set)}")
    print(f"Number of Batches: {len(data_loader)}")

    return data_set, data_loader
