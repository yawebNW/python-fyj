import numpy as np
import pandas as pd
from numpy.ma.extras import hstack

# array1 = np.arange(10);
# print(array1)
# print(array1.ndim)
# print(array1.shape)
# print(type(array1))
#
# print(np.sqrt(array1))
# print(np.exp(array1))
# print(np.sum(array1))
# print(np.mean(array1))
#
# print(np.zeros(3))
# print(np.zeros((3,3)))
# print(np.ones((3)))
# print(np.ones((3,3)))
#
# array1 = np.random.randint(1,10,(2,4))
# print(array1)
# print(array1*2+1)
# print(array1**2)
# print(np.sum(array1))
# print(np.mean(array1))
# print(array1.mean(axis=0))
# print(array1.mean(axis=1))
# print(array1.cumsum(axis=0))
# print(array1.cumsum(axis=1))
# print(array1.cumsum())
#
# list1 = [[1,1],[2,3]]
# array1 = np.array(list1)
# print(array1)
# print(array1.T)
# print(np.dot(array1,array1.T))
# print(np.linalg.inv(array1))
# print(np.linalg.eig(array1))
#
# array1 = np.random.randint(1,100,(4,6))
# print(array1)
# array1.sort()
# print(array1)

# array1 = np.arange(24).reshape((4,6))
# print(array1)
# array1.sort(axis=0)
# print(array1)
# array1.sort(axis=1)
# print(array1)

# print(np.unique(array1))
# print(np.where(array1 > 5,100,array1))

# mask = np.array([[0,0,1,1,1,0],
#                  [0,1,1,0,1,0],Z
#                  [1,1,0,0,1,0]], dtype=bool)
# print(array1[mask])
#
# print(array1[[0,2,1]])
# print(array1[[0,2,2,1],[1,2,3,0]])
# print(array1[np.ix_([0,2],[1,2,3,4])])
# print(np.hsplit(array1,2))
# print(np.vsplit(array1,2))
# print(np.split(array1,2,axis=0))
# print(np.split(array1,2,axis=1))
# print(np.sort(array1))
# print(np.sort(array1, axis=0))
# print(np.sort(array1, axis=1))
# sort_ind = np.argsort(array1[:,2])
# print(sort_ind)
# print(array1[sort_ind])

# 创建两个排序键的数组
# array1 = np.random.randint(50,100,(6,3))
# array2 = array1.sum(axis=1)
# print(array1)
# print(array2)
# array1 = np.hstack((array1,array2.reshape(-1,1)))
# print(array1[1:3])
# print(np.argsort(array1[:,3]))
# print(array1[np.argsort(array1[:,3])[::-1]])
# print(array1[np.lexsort((array1[:,2],array1[:,1],array1[:,0],array1[:,3]))[::-1]])
# print(np.argmax(array1[:,3]))
# print(np.argmin(array1[:,3]))
# print('总成绩平均值为%d'%array1[:,3].mean())
# morethanmean = np.where(array1[:,3] > array1[:,3].mean(),True,False )
# print(morethanmean)
# print(array1[morethanmean])
# print(array1[1:3] > 80)
# print(np.extract(array1[1:3] > 80, array1))

# s1 = pd.Series(np.arange(20),['a','b','c','d','e'], name='s1')
# print(s1)
# print(type(s1))

d1 = {'语文':[60,80,76,89],'数学':[50,27,78,49],'英语':[80,79,39,69]}
data = pd.DataFrame(d1)
print(data)
print(data.columns)
print(data.values)
print(data.axes)
print(data.ndim)
print(data.size)
print(data.shape)
print(data.head(2))
print(data.tail(2))
print(data.drop('英语',axis=1))