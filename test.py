import numpy as np
def sig(x):
    return 1/(1+np.exp(-x))
lst=np.array([i+1 for i in range(28*28)])
lst1=lst.copy()
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
for i in range(7):
    for j in range(len(cc1)):
        lst1[kk]=lst[28*i+cc1[j]]
        kk+=1
    for j in range(len(cc2)):
        lst1[kk]=lst[cc2[j]]
        kk+=1
print(cc1)
print(cc2)
print(lst1)