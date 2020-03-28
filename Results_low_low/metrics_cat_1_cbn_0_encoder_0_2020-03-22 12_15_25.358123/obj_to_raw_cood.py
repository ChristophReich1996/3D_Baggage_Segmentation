import numpy as np

with open('25_outfile_label.obj') as file:
    lines = file.readlines()

data = []

for line in lines:
    x, y, z = line.split(' ')[2:5]
    data.append([float(x), float(y), float(z)])

data = np.array(data)
data = data[:-8]
data = data[np.random.choice(data.shape[0], int(data.shape[0] // 15), replace=False)]

np.savetxt('25_outfile_label.txt', data, delimiter=' ', fmt='%1.3f')