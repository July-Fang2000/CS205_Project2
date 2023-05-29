import numpy as np


# Define a function to read data from a .txt file
def read_txt(filename):
    # Initialize empty lists for data and labels
    datas = []
    labels = []
    # Open the file for reading
    with open("./dataset/" + filename, 'r') as file:
        # Loop through each line in the file
        for line in file:
            # Split the line into a list of values
            data = line.split()
            # The first value is the label, append it to the labels list
            labels.append(float(data[0]))
            # The rest of the values are the features, append them to the datas list
            datas.append([float(x) for x in data[1:]])
    # Return the data and labels
    return datas, labels


# cite: https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/normalize-data?view=azureml-api-2
# def normalized_data(data):
#     datas_normalized = 2 * (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)) - 1
#     return datas_normalized

# Define a function to standardize the data
def standard_data(data):
    # Subtract the mean and divide by the standard deviation
    datas_standard = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    # Return the standardized data
    return datas_standard
