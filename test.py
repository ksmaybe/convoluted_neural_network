import numpy as np
def sig(x):
    return 1/(1+np.exp(-x))
lst=np.array([i+1 for i in range(784)])
lst=lst.reshape((196,4))
lst1=sig(np.sum(lst,axis=1))
print(lst1)
