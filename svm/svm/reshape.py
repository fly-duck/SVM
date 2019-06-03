import numpy as np 
a = np.arange(6).reshape((3, 2))
print(a)
a = np.arange(6).reshape((1, 6))
print(a)
#-1就是根据3而适应，因为原来总共有6个数据，-1就代表 6/3=2
a = np.arange(6).reshape((-1, 3))
print(a)
#-1就是根据2而适应，因为原来总共有6个数据，-1就代表 6/2=3
a = np.arange(6).reshape((-1, 2))
print(a)

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
# >>> a = np.array([0, 1, 2])
# >>> np.tile(a, 2)
# array([0, 1, 2, 0, 1, 2])
# >>> np.tile(a, (2, 2))
# array([[0, 1, 2, 0, 1, 2],
#        [0, 1, 2, 0, 1, 2]])
# >>> np.tile(a, (2, 1, 2))
# array([[[0, 1, 2, 0, 1, 2]],
#        [[0, 1, 2, 0, 1, 2]]])