import sys
import torch
import numpy as np
import torch.utils.data as Data


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, num):
    data = []
    label = []
    start_index = start_index + history_size*(5-int(num))
    if end_index is None:
        end_index = len(dataset)-target_size

    for i in range(start_index, end_index-history_size*(int(num)-1), history_size):
        indices = range(i-96, i, step)#468
        data.append(dataset[indices])
    label = target[4:]
    return torch.tensor(np.array(data)), torch.tensor(np.array(label))

def return_dataset(dataset10,dataset20,dataset30,dataset40, label):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    datasets = [dataset10.values, dataset20.values, dataset30.values, dataset40.values]
    label = label.values

    label = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5) / 5, mode='same'), axis=0, arr=label)
    for i in range(len(datasets)):
        datasets[i] = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5) / 5, mode='same'), axis=0, arr=datasets[i])

    for i, dataset in enumerate(datasets):
        timevalue = 24*(i+1)
        num = timevalue/24
        iny = 124
        splity = 470
        end = splity + iny
        splitx = splity*timevalue
        endx = end*timevalue
        iny = iny*timevalue

        past_history = timevalue
        future_target = 1
        STEP = 1
        
        x_train_temp, y_train_temp = multivariate_data(
            dataset[:splitx,:], label[:splity], 0, splitx, past_history, future_target, STEP, num)
        x_train_temp, y_train_temp = x_train_temp.squeeze(-1),y_train_temp.squeeze(-1)
        x_val_temp, y_val_temp = multivariate_data(
            dataset[splitx:endx,:], label[splity:end], 0, iny, past_history, future_target, STEP, num)
        x_val_temp, y_val_temp = x_val_temp.squeeze(-1), y_val_temp.squeeze(-1)

        x_train.append(x_train_temp)
        y_train.append(y_train_temp)
        x_val.append(x_val_temp)
        y_val.append(y_val_temp)

    return x_train, y_train, x_val, y_val

def dataset_read36(df6,df12,df18,df24, label, batch_size):

    x_train, y_train, x_val, y_val = return_dataset(df6,df12,df18,df24, label)

    datasets = []
    data_loaders = []

    for i in range(4):
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
