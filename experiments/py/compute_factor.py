import numpy as np
data = np.loadtxt('time_var_torus.txt')
time = data[:,1]
var = data[:,2]
factor = np.sqrt(time[1:]/time[:-1] * var[:-1] / var[1:])
print(factor)
base = 8
deg = 6
samples_0 = []
num_samples = base
samples_0.append(num_samples)
for i in range(deg):
    num_samples = np.ceil(num_samples * factor[deg-1-i])
    samples_0.append(num_samples)
samples_0.reverse()
print(samples_0)

samples_1 = []
num_samples = base
samples_1.append(num_samples)
for i in range(deg-1):
    num_samples = np.ceil(num_samples * factor[deg-2-i])
    samples_1.append(num_samples)
samples_1.reverse()
print(samples_1)
