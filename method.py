import numpy as np


# Define the Nearest Neighbor Classifier
class NearestNeighborClassifier:
    def __init__(self, datas, labels):
        # Initialize the classifier with data and labels
        self.datas = datas
        self.labels = labels

    @staticmethod
    def calculate_distances(data, datas):
        # Compute and return the Euclidean distance between data points
        return np.sum((datas - data) ** 2, axis=1)

    def find_nearest(self, index):
        # Compute distances from the given data point to all others
        # Replace the distance to itself with infinity to avoid picking itself
        # Return the index of the nearest data point
        distances = self.calculate_distances(self.datas[index], self.datas)
        distances[index] = float('inf')
        return np.argmin(distances)

    def predict(self, index):
        # Predict the label of the given data point by looking at its nearest neighbor
        return self.labels[self.find_nearest(index)]

    def accuracy(self):
        # Compute and return the accuracy of the classifier
        correct = 0
        for i in range(len(self.datas)):
            if self.labels[i] == self.predict(i):
                correct += 1
        return round((correct / len(self.datas)) * 100, 1)

    def accuracy_with_features(self, features):
        # Compute and return the accuracy of the classifier when only a subset of features is used
        datas_sub = self.datas[:, features]
        sub_classifier = NearestNeighborClassifier(datas_sub, self.labels)
        return sub_classifier.accuracy()


def search(method, classifier):
    # Depending on the method chosen by the user, call the appropriate feature selection function
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
    # This function generates all possible combinations of feature sets by adding or removing one feature at a time.
    # 'feature_set' is the current set of features,
    # 'size' is the total number of features,
    # 'method' is the selection method ('forward' or 'backward').

    explored_sets = []  # Initialize an empty list to store all explored feature sets

    if method == "forward":
        # In forward selection, we add one feature at a time
        for i in range(size):
            cur_set = list(feature_set)  # Create a copy of the current feature set
            if i not in cur_set:  # If the feature is not already in the set
                cur_set.append(i)  # Add it
                explored_sets.append(cur_set)  # Add the new set to the list of explored sets
    elif method == "backward":
        # In backward selection, we remove one feature at a time
        for i in range(size):
            cur_set = list(feature_set)  # Create a copy of the current feature set
            if i in cur_set:  # If the feature is in the set
                cur_set.remove(i)  # Remove it
                explored_sets.append(cur_set)  # Add the new set to the list of explored sets
    else:
        raise ValueError("Invalid method. Choose 'forward' or 'backward'.")

    return explored_sets  # Return all explored feature sets


def feature_selection(classifier, size, method):
    # This function performs the actual feature selection, based on the chosen method (forward or backward).
    # 'classifier' is the classifier used for evaluating feature subsets,
    # 'size' is the total number of features,
    # 'method' is the selection method ('forward' or 'backward').

    # Check if the method is valid
    if method not in ['forward', 'backward']:
        raise ValueError("Invalid method. Choose 'forward' or 'backward'.")

    max_accuracy = -2  # Initialize the maximum accuracy to a very low value
    feature_set = list(range(size)) if method == 'backward' else []  # Initialize the feature set based on the method

    # Loop through each feature
    for i in range(size):
        curr_accuracy = -1  # Initialize the current maximum accuracy to a very low value
        sets = explore_features(feature_set, size, method)  # Get all possible feature sets for this iteration

        # Loop through each possible feature set
        for expanded in sets:
            accuracy = classifier.accuracy_with_features(expanded)  # Compute the accuracy for this feature set
            print("Using feature(s) {", ','.join(map(str, [feature + 1 for feature in expanded])), "} accuracy is",
                  accuracy, "%")
            if accuracy > curr_accuracy:  # If this feature set is better than the current best
                curr_accuracy = accuracy  # Update the current maximum accuracy
                feature_set = expanded  # Update the current best feature set

        if curr_accuracy > max_accuracy:  # If the current best feature set is better than the overall best
            max_accuracy = curr_accuracy  # Update the maximum accuracy
            max_set = feature_set  # Update the best feature set
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        print("Feature set {", ','.join(map(str, [feature + 1 for feature in feature_set])), "} was best, accuracy is",
              curr_accuracy, "%\n")

    # Print the best feature set and its accuracy
    print("Finished search!! The best feature subset is {", ','.join(map(str, [feature + 1 for feature in max_set])),
          "}, which has an accuracy of", max_accuracy, "%\n")
