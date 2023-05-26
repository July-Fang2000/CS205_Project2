import numpy as np


class NearestNeighborClassifier:
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    @staticmethod
    def calculate_distance(data1, data2):
        return np.sum((data1 - data2) ** 2)

    def find_nearest(self, index):
        min_distance = float('inf')
        nearest_index = -1
        for i, data in enumerate(self.datas):
            if i == index:
                continue
            distance = self.calculate_distance(data, self.datas[index])
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        return nearest_index

    def predict(self, index):
        return self.labels[self.find_nearest(index)]

    def accuracy(self):
        correct = 0
        for i in range(len(self.datas)):
            if self.labels[i] == self.predict(i):
                correct += 1
        return (correct / len(self.datas)) * 100
