import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])

b = np.array([[3,2,1],[6,5,4],[9,8,7]])

c = np.insert(a,[0,5,1],b,axis=0)

print(c)