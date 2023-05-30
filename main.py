import pandas as pd
import os
from preprocess import *
from method import *

print("Welcome to Zelai Fang and Hengshuo Zhang Feature Selection Algorithm.")

# Get the file name from the user
filename = input("Type in the name of the file to test: ")

# Check the file extension of the input file
_, file_extension = os.path.splitext(filename)

# Get the algorithm choice from the user
method = input("Type the number of the algorithm you want to run.\n"
               "     1) Forward Selection\n"
               "     2) Backward Elimination\n")

# Initialize empty lists for data and labels
datas = []
labels = []

# Read the data based on its file extension
if file_extension == ".txt":
    # If it's a .txt file, use read_txt function from preprocess.py
    datas, labels = read_txt(filename)
elif file_extension == ".csv":
    # If it's a .csv file, use pandas to read the csv file
    # Drop the first and the last column, and map 'M' and 'B' to 1 and 2 respectively in the first column
    data = pd.read_csv("./dataset/"+filename)
    data = data.drop(data.columns[[0, -1]], axis=1)
    data[data.columns[0]] = data[data.columns[0]].map({'M': 1, 'B': 2})
    data = data.values
    labels = data[:, 0]
    datas = data[:, 1:]

# Standardize the data
datas = standard_data(datas)

# Instantiate a Nearest Neighbor Classifier with the data and labels
classifier = NearestNeighborClassifier(datas, labels)

# Perform the feature selection based on the user's choice of method
search(method, classifier)
