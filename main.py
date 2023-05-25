from preprocess import *
from method import *

print("Welcome to Zelai Fang and Hengshuo Zhang Feature Selection Algorithm.")
filename = input("Type in the name of the file to test: ")
# method = input("Type the number of the algorithm you want to run.\n"
#                "     1) Forward Selection\n"
#                "     2) Backward Selection\n")

datas, labels = read_txt(filename)
feature_size = len(datas[0])
datas = normalized_data(datas)

# print(feature_size)
# print(datas)
# print(labels)
