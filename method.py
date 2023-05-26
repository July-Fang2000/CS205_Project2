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


def search(method, feature_size, instance_size, accuracy):
    print("This dataset has {} features(not including the class attribute), with {} instances."
          .format(feature_size, instance_size))
    print("Running nearest neighbor with all {} features, "
          "using \"leaving-one-out\" evaluation, I get an accuracy of {}%".format(feature_size, accuracy))
    print("Beginning search.")
    if method == "1":
        forward_selection()
    elif method == "2":
        backward_elimination()
    else:
        print("Choose Wrong Method! Please type 1 or 2!")


def forward_selection():
    return 0


def backward_elimination():
    return 0
