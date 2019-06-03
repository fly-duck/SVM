import numpy as np
from sklearn import svm
import  matplotlib.pyplot as plt

np.random.seed(0)
x = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
print(x)
y = [-1]*20+[1]*20
print(y)
clf = svm.SVC(kernel = 'linear')#使用线性和
clf.fit(x,y)
w = clf.coef_[0]#获取w
print(w)
a = -w[0]/w[1]#获取斜率

#画图
xx = np.linspace(-5,5)
yy = a*xx - (clf.intercept_[0])/w[1]
b = clf.support_vectors_[0]
yy_down = a*xx+(b[1]-a*b[0])
b = clf.support_vectors_[-1]
print(b)
yy_up = a*xx+(b[1]-a*b[0])
plt.figure(figsize=(8,4))
plt.plot(xx,yy)
plt.plot(xx,yy_down)
plt.plot(xx,yy_up)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s =80)
plt.scatter(x[:,0],x[:,1],c = y,cmap=plt.cm.Paired)
plt.axis('tight')
plt.show()




#np.r_按列连接两个矩阵，就是把两个矩阵上下相加，要求列数相等
#np.c_按行连接连个举证，就是把两个举证左右相加，要求行数相等