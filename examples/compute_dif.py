import numpy as np
arr = np.genfromtxt("ref_pot.csv", delimiter=",")
arr_ = np.genfromtxt("pot.csv", delimiter=",")
dif = arr - arr_
dif[:,:3] = arr[:,:3]
np.savetxt('pot_dif.csv', dif, delimiter=',')
print(dif)
