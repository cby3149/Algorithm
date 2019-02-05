import numpy as np
import scipy.io as scio
theta = np.loadtxt('test.csv', delimiter = ',')
data = scio.loadmat('/Users/chenboyuan/Documents/machine-learning-ex/ex3/ex3data1.mat')
X = data.get('X')
y = data.get('y')
X = np.c_[np.ones(X.shape[0]).T,X]
m = X.shape[0]
n = X.shape[1]
pred = []
count = 0
for i in range(m):
     temp = np.dot(theta,X[i,:].T)
     maxi = max(temp)
     index = list(temp).index(maxi)
     pred.append(index+1)

for each in range(5000):
    if pred[each]==list(y.reshape([5000,]))[each]:
        count += 1
print(count/5000)
# 84.68%