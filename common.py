def read_txt(filename):
    datas = []
    with open("./dataset/"+filename, 'r') as file:
        for line in file:
            data = line.split()
            datas.append(data)
        return datas
