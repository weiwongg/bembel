import numpy as np
f = open('deformed_gamma_3.dat')
f_out = open('deformed_gamma_.dat','w')
for i in range(5):
    f_out.write(f.readline())
num_patches = 18
dim = 5
scale = 1
for i in range(num_patches):
    for j in range(5):
        f_out.write(f.readline())
    x = np.array([float(item) for item in f.readline().strip().split('   ')])
    x = scale*np.flip(x.reshape(dim,dim),1).flatten()
    f_out.write("   ".join(str(item) for item in x) + '\n')
    y = np.array([float(item) for item in f.readline().strip().split('   ')])
    y = scale*np.flip(y.reshape(dim,dim),1).flatten()
    f_out.write("   ".join(str(item) for item in y) + '\n')
    z = np.array([float(item) for item in f.readline().strip().split('   ')])
    z = scale*np.flip(z.reshape(dim,dim),1).flatten()
    f_out.write("   ".join(str(item) for item in z) + '\n')
    w = np.array([float(item) for item in f.readline().strip().split('   ')])
    w = np.flip(w.reshape(dim,dim),1).flatten()
    f_out.write("   ".join(str(item) for item in w) + '\n')
f_out.close()
