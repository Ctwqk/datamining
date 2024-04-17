import numpy as np
npdata=np.random.randn(10000,5)


centroids=np.random.randn(5,10)

print(abs(np.expand_dims(npdata[200],1)-centroids).sum(axis=0))

d=np.expand_dims(npdata[200], 1)


a=np.array([1,2,3,4,5])
b=np.array([[2,4,6,8,10],[1,3,5,7,9]])
print(b-a)



x=[1,1,2,1,3,4,1,2,3,1,2,3,4]
b=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
print(b[x==1])