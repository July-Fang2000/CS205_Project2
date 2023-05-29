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
        return round((correct / len(self.datas)) * 100, 1)

    def accuracy_with_features(self, features):
        datas_sub = self.datas[:, features]
        sub_classifier = NearestNeighborClassifier(datas_sub, self.labels)
        return sub_classifier.accuracy()


def search(method, classifier):
    feature_size = len(classifier.datas[0])
    instance_size = len(classifier.datas)
    print("This dataset has {} features(not including the class attribute), with {} instances."
          .format(feature_size, instance_size))
    print("Running nearest neighbor with all {} features, "
          "using \"leaving-one-out\" evaluation, I get an accuracy of {}%"
          .format(feature_size, classifier.accuracy()))
    print("Beginning search.")
    if method == "1":
        feature_selection(classifier, feature_size, method="forward")
    elif method == "2":
        feature_selection(classifier, feature_size, method="backward")
    else:
        print("Choose Wrong Method! Please type 1 or 2!")


def explore_features(feature_set, size, method):
    explored_sets = []
    if method == "forward":
        for i in range(size):
            cur_set = list(feature_set)
            if i not in cur_set:
                cur_set.append(i)
                explored_sets.append(cur_set)
    elif method == "backward":
        for i in range(size):
            cur_set = list(feature_set)
            if i in cur_set:
                cur_set.remove(i)
                explored_sets.append(cur_set)
    else:
        raise ValueError("Invalid method. Choose 'forward' or 'backward'.")
    return explored_sets


def feature_selection(classifier, size, method):
    if method not in ['forward', 'backward']:
        raise ValueError("Invalid method. Choose 'forward' or 'backward'.")

    max_accuracy = -2
    feature_set = list(range(size)) if method == 'backward' else []

    for i in range(size):
        curr_accuracy = -1
        sets = explore_features(feature_set, size, method)
        for expanded in sets:
            accuracy = classifier.accuracy_with_features(expanded)
            print("Using feature(s) {", ','.join(map(str, [feature + 1 for feature in expanded])), "} accuracy is",
                  accuracy, "%")
            if accuracy > curr_accuracy:
                curr_accuracy = accuracy
                feature_set = expanded
        print("")
        if curr_accuracy > max_accuracy:
            max_accuracy = curr_accuracy
            max_set = feature_set
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        print("Feature set {", ','.join(map(str, [feature + 1 for feature in feature_set])), "} was best, accuracy is",
              curr_accuracy, "%\n")

    print("Finished search!! The best feature subset is {", ','.join(map(str, [feature + 1 for feature in max_set])),
          "}, which has an accuracy of", max_accuracy, "%\n")
