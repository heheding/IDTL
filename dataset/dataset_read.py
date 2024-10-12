import sys
import torch
import numpy as np
import torch.utils.data as Data


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)-target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return torch.tensor(np.array(data)), torch.tensor(np.array(labels))

def return_dataset(dataset10,dataset20,dataset30,dataset40,dataset50,dataset60, label):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    datasets = [dataset10.values, dataset20.values, dataset30.values, dataset40.values, dataset50.values, dataset60.values]
    label = label.values

    label = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5) / 5, mode='same'), axis=0, arr=label)
    for i in range(len(datasets)):
        datasets[i] = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5) / 5, mode='same'), axis=0, arr=datasets[i])

    for i, dataset in enumerate(datasets):
        timevalue = 78*(i+1)
        num = timevalue/78
        iny = 400
        splity = 1000
        end = splity + iny
        splitx = splity*timevalue
        endx = end*timevalue
        iny = iny*timevalue
        train_min = dataset[:].min(0)
        train_max = dataset[:].max(0)
        train_cha = train_max - train_min
        dataset = (dataset-train_min)/(train_cha+0.000001)

        past_history = timevalue
        future_target = 1
        STEP = 1
        
        x_train_temp, y_train_temp = multivariate_data(
            dataset[:splitx,:15], label[:splity], 0, splitx, past_history, future_target, STEP, num)
        x_train_temp, y_train_temp = x_train_temp.squeeze(-1),y_train_temp.squeeze(-1)
        x_val_temp, y_val_temp = multivariate_data(
            dataset[splitx:endx,:15], label[splity:end], 0, iny, past_history, future_target, STEP, num)
        x_val_temp, y_val_temp = x_val_temp.squeeze(-1), y_val_temp.squeeze(-1)

        x_train.append(x_train_temp)
        y_train.append(y_train_temp)
        x_val.append(x_val_temp)
        y_val.append(y_val_temp)

    return x_train, y_train, x_val, y_val

def dataset_read(df6,df12,df18,df24,df30,df36, label, batch_size):

    x_train, y_train, x_val, y_val = return_dataset(df6,df12,df18,df24,df30,df36, label)

    datasets = []
    data_loaders = []

    for i in range(6):
        dataset = torch.utils.data.TensorDataset(x_train[i], y_train[i])
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )
        datasets.append(dataset)
        data_loaders.append(data_loader)

    return data_loaders, x_val, y_val

def dataset_readcnn(df6,  batch_size):

    x_train, y_train, x_val, y_val = return_datasetcnn(df6)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )


    return data_loader, x_val, y_val

def return_datasetcnn(df):
    features = df
    features = features.rolling(window=5).mean()
    features.drop(features.head(5).index, inplace=True)

    dataset = features.values
    train_split = 480

    past_history = 30
    future_target = 1
    STEP = 1

    x_train, y_train = multivariate_data(
        dataset[:, :], dataset[:, -1], 0, train_split, past_history, future_target, STEP, single_step=True)
    x_val, y_val = multivariate_data(
        dataset[:, :], dataset[:, -1], train_split, None, past_history, future_target, STEP, single_step=True)


    # return train_mean, train_std, x_train, y_train, x_val, y_val
    return  x_train, y_train, x_val, y_val
