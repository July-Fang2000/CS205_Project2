import numpy as np


class NearestNeighborClassifier:
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    @staticmethod
    def calculate_distances(data, datas):
        return np.sum((datas - data) ** 2, axis=1)

    def find_nearest(self, index):
        distances = self.calculate_distances(self.datas[index], self.datas)
        distances[index] = float('inf')
        return np.argmin(distances)

    def predict(self, index):
        return self.labels[self.find_nearest(index)]

    def accuracy(self):
        correct = 0
        for i in range(len(self.datas)):
            if self.labels[i] == self.predict(i):
                correct += 1
        return (correct / len(self.datas)) * 100
