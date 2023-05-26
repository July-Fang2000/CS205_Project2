from preprocess import *
from method import *

print("Welcome to Zelai Fang and Hengshuo Zhang Feature Selection Algorithm.")
filename = input("Type in the name of the file to test: ")
method = input("Type the number of the algorithm you want to run.\n"
               "     1) Forward Selection\n"
               "     2) Backward Selection\n")

datas, labels = read_txt(filename)
feature_size = len(datas[0])
instance_size = len(datas)
datas = normalized_data(datas)

classifier = NearestNeighborClassifier(datas, labels)
accuracy = classifier.accuracy()

print("This dataset has {} features(not including the class attribute), with {} instances."
      .format(feature_size, instance_size))
print("Running nearest neighbor with all {} features, "
      "using \"leaving-one-out\" evaluation, I get an accuracy of {}%".format(feature_size, accuracy))
print("Beginning search.")
