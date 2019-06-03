from scipy import stats 
import numpy as np 
from scipy.stats import multivariate_normal

mean=[0,0]
# cov=np.eye(2)
# 多元高斯分布，或多元正太分布，数据的分布跟协方差矩阵有关

cov=[[1,0],[0,100]]


import matplotlib.pyplot as plt 

mn=stats.multivariate_normal(mean,cov)
print(mn)

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal
#这个cov必须是半正定矩阵
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))
# rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
rv=multivariate_normal(mean,cov)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, rv.pdf(pos))
plt.show()

# print()