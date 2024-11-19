from LTSF.data_provider1.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
import pandas as pd
from DataProviders.LoadForecasting import LoadForecastingCustomDataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'load_forecasting_custom': LoadForecastingCustomDataset,  # Custom dataset for load forecasting
    'combinedDaytonData_fill': LoadForecastingCustomDataset   
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'load_forecasting_custom':
        # Load and prepare custom dataset without additional arguments
        df = pd.read_csv(args.data_path, parse_dates=['datetime'])
        df = df[['temperature', 'precipitation', 'humidity', 'wind_speed', 'dayton_mw']]
        data_set = LoadForecastingCustomDataset(df, look_back=args.seq_len)
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            train_only=train_only
        )

    print(f"{flag} set size: {len(data_set)}")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
