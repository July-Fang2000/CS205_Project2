import numpy as np


def read_txt(filename):
    datas = []
    labels = []
    with open("./dataset/" + filename, 'r') as file:
        for line in file:
            data = line.split()
            labels.append(data[0])
            datas.append([float(x) for x in data[1:]])
        return datas, labels


def normalized_data(datas):
    datas = np.array(datas)
    datas_normalized = (datas - np.min(datas, axis=0)) / (np.max(datas, axis=0) - np.min(datas, axis=0))
    return datas_normalized
