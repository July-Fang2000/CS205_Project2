import numpy as np


def read_txt(filename):
    datas = []
    labels = []
    with open("./dataset/" + filename, 'r') as file:
        for line in file:
            data = line.split()
            labels.append(float(data[0]))
            datas.append([float(x) for x in data[1:]])
    return datas, labels


# cite: https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/normalize-data?view=azureml-api-2
def normalized_data(data):
    datas_normalized = 2 * (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)) - 1
    return datas_normalized


def standard_data(data):
    datas_standard = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return datas_standard
