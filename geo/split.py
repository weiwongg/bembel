import numpy as np
f = open('cube.dat')
f_out = open('refined_cube.dat','w')
for i in range(5):
    f_out.write(f.readline())
num_patches = 6
dim = 2
scale = 1
patch_id = 0
for i in range(num_patches):
    for j in range(3):
        f.readline()
    ctr_u = np.array([float(item) for item in f.readline().strip().split('   ')])
    ctr_u = ctr_u.flatten()
    ctr_v = np.array([float(item) for item in f.readline().strip().split('   ')])
    ctr_v = ctr_v.flatten()
    x = np.array([float(item) for item in f.readline().strip().split('   ')])
    x = x.reshape(dim,dim)
    x = np.insert(x,[1],np.mean(x,1)[:, np.newaxis],axis=1)
    x = np.insert(x,[1],np.mean(x,0)[np.newaxis,:],axis=0)
    y = np.array([float(item) for item in f.readline().strip().split('   ')])
    y = y.reshape(dim,dim)
    y = np.insert(y,[1],np.mean(y,1)[:, np.newaxis],axis=1)
    y = np.insert(y,[1],np.mean(y,0)[np.newaxis,:],axis=0)
    z = np.array([float(item) for item in f.readline().strip().split('   ')])
    z = z.reshape(dim,dim)
    z = np.insert(z,[1],np.mean(z,1)[:, np.newaxis],axis=1)
    z = np.insert(z,[1],np.mean(z,0)[np.newaxis,:],axis=0)
    w = np.array([float(item) for item in f.readline().strip().split('   ')])
    w = w.flatten()
    for ii in range(2):
        for jj in range(2):
            f_out.write(f"PATCH {patch_id}" + '\n')
            f_out.write("1 1" + '\n')
            f_out.write("2 2" + '\n')
            f_out.write("   ".join(str(item) for item in ctr_u) + '\n')
            f_out.write("   ".join(str(item) for item in ctr_v) + '\n')
            xx = x[ii:ii+2,jj:jj+2].flatten()
            yy = y[ii:ii+2,jj:jj+2].flatten()
            zz = z[ii:ii+2,jj:jj+2].flatten()
            f_out.write("   ".join(str(item) for item in xx) + '\n')
            f_out.write("   ".join(str(item) for item in yy) + '\n')
            f_out.write("   ".join(str(item) for item in zz) + '\n')
            f_out.write("   ".join(str(item) for item in w) + '\n')
            patch_id = patch_id + 1
f_out.close()
