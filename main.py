import pandas as pd
import os
from preprocess import *
from method import *

print("Welcome to Zelai Fang and Hengshuo Zhang Feature Selection Algorithm.")

filename = input("Type in the name of the file to test: ")
# if filename == "1":
#     filename = "CS170_small_Data__32.txt"
# elif filename == "2":
#     filename = "CS170_small_Data__33.txt"
# elif filename == "3":
#     filename = "CS170_large_Data__32.txt"
# elif filename == "4":
#     filename = "CS170_large_Data__33.txt"
_, file_extension = os.path.splitext(filename)

method = input("Type the number of the algorithm you want to run.\n"
               "     1) Forward Selection\n"
               "     2) Backward Selection\n")

if file_extension == ".txt":
    datas, labels = read_txt(filename)
elif file_extension == ".csv":
    data = pd.read_csv("./dataset/"+filename)
    data = data.drop(data.columns[[0, -1]], axis=1)
    data[data.columns[0]] = data[data.columns[0]].map({'M': 1, 'B': 2})
    data = data.values
    labels = data[:, 0]
    datas = data[:, 1:]

datas = standard_data(datas)

classifier = NearestNeighborClassifier(datas, labels)

search(method, classifier)
