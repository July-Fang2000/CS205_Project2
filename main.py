from preprocess import *
from method import *

print("Welcome to Zelai Fang and Hengshuo Zhang Feature Selection Algorithm.")
filename = input("Type in the name of the file to test: ")
if filename == "1":
    filename = "CS170_small_Data__32.txt"
elif filename == "2":
    filename = "CS170_small_Data__33.txt"
elif filename == "3":
    filename = "CS170_large_Data__32.txt"
elif filename == "4":
    filename = "CS170_large_Data__33.txt"

method = input("Type the number of the algorithm you want to run.\n"
               "     1) Forward Selection\n"
               "     2) Backward Selection\n")

datas, labels = read_txt(filename)
feature_size = len(datas[0])
instance_size = len(datas)
datas = normalized_data(datas)
classifier = NearestNeighborClassifier(datas, labels)
search(method, feature_size, instance_size, classifier)
