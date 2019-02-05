import scipy.io as scio
import numpy as np

data = scio.loadmat('/Users/chenboyuan/Documents/machine-learning-ex/ex3/ex3data1.mat')
X = data.get('X')
y = data.get('y')

def sigmoid(x):
    """
    :param x: Matrix
    :return: 1/(1 + e^(- x))
    """
    return 1/(1 + np.exp(-x))

def J(X,y,theta,lam):
    """
    :param X: X Matrix
    :param y: y Matrix
    :param theta: theta
    :param lam: lambda
    :return: J(theta) / Cost Function
    """
    m = X.shape[0]
    temp = sigmoid(np.dot(X,theta))-0.0000000000000001
    return -1*1/(m)*(np.dot(y.T,np.log(temp))+np.dot((1-y).T,np.log(1-temp)))+lam/(2*m)*np.sum(theta[1:m]*theta[1:m])

def Grad(X,y,theta,lam,alpha):
    """
    :param X: X Matrix
    :param y: y Matrix
    :param theta: theta
    :param lam: lambda
    :param alpha: alpha
    :return: Gradient Descent
    """
    m = X.shape[0]
    n = X.shape[1]
    temp = sigmoid(np.dot(X, theta)) - 0.0000000000000001
    temp_2 = theta
    temp_2[0] = 0
    return theta.reshape([n,1]) - alpha*(1/m*np.dot(X.T,temp.reshape([m,1])-y).reshape([n,1])) + lam/m*theta.reshape([n,1]).reshape([n,1])


X = np.c_[np.ones(X.shape[0]).T,X]
theta = np.ones(X.shape[1]).T/3
final_theta = np.ones([10,X.shape[1]])
lam = 2
alpha = 0.01

count = 0
for i in range(1,11):
    temp = y.copy()
    temp = temp.reshape([5000,])
    temp[temp != i] =0
    temp[temp == i] = 1
    theta = np.ones(X.shape[1]).T / 3
    temp_1 = 0
    temp_2 = 1
    temp = np.array(temp).reshape([5000,1])

    while abs(temp_1 - temp_2) > 0.00001:
        count += 1
        temp_1 = J(X,temp,theta,lam)
        theta = Grad(X,temp,theta,lam,alpha)
        temp_2 = J(X,temp,theta,lam)
        print(abs(temp_1 - temp_2),temp_2,'计算次数为: ', count,' 当前拟合为: ', i)
    final_theta[i-1,:] = theta.reshape([401,])
    count = 0

np.savetxt('test.csv',final_theta,delimiter=',')