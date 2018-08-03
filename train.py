import tensorflow as tf 
import numpy as np 
a = np.array([[1,2,3,4],[3,4,5,6],[5,6,7,8]])
b = np.array([[11,22,33,44],[33,45,67,88],[34,36,26,57]])
c = np.zeros((3,4))
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		if a[i][j]>b[i][j]:
			c[i][j] = a[i][j]
		else:
			c[i][j] = b[i][j]


print(c)

