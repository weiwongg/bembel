import numpy as np
f = open('sphere.dat')
f_out = open('artifical_sphere.dat','w')
for i in range(5):
    f_out.write(f.readline())
num_patches = 6
dim = 5
scale = 3
for i in range(num_patches):
    for j in range(5):
        f_out.write(f.readline())
    x = np.array([float(item) for item in f.readline().strip().split('   ')])
    x = scale*x
    f_out.write("   ".join(str(item) for item in x) + '\n')
    y = np.array([float(item) for item in f.readline().strip().split('   ')])
    y = scale*y
    f_out.write("   ".join(str(item) for item in y) + '\n')
    z = np.array([float(item) for item in f.readline().strip().split('   ')])
    z = scale*z
    f_out.write("   ".join(str(item) for item in z) + '\n')
    w = np.array([float(item) for item in f.readline().strip().split('   ')])
    f_out.write("   ".join(str(item) for item in w) + '\n')
f_out.close()
