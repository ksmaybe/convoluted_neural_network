import numpy as np
import copy
def sig(x):
    return 1/(1+np.exp(-x))
lst=np.array([i for i in range(28*28)])
lst1=copy.deepcopy(lst)
cc1=[]
cc2=[]
x=0
for i in range(14):
    cc1.append(x)
    x+=1
    cc1.append(x)
    x+=1
    cc2.append(x)
    x+=1
    cc2.append(x)
    x+=1
kk=0
for i in range(14):
    for j in range(len(cc1)):
        lst1[kk]=lst[56*i+cc1[j]]
        kk+=1
    for j in range(len(cc2)):
        lst1[kk]=lst[56*i+cc2[j]]
        kk+=1

print(lst1.reshape((28,28)))