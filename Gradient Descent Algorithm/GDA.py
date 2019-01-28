import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('ex1data1.txt', delimiter = ',')
row,col = data.shape
for i in range(col):
    data[i] = (data[i] - data[i].mean())/(data[i].max()-data[i].min())
data = np.c_[np.ones(row).T,data]
"""
Parameters X, y, theta, iter, alpha
"""
X = data[:,0:2]
y = data[:,2]
theta = np.array([2,2]).T
iter = 10000
alpha = 0.01


def computeJ(X,y,theta):
    """
    Compute J(theta)
    1/(2*row)*(X*theta - y).^2
    """
    J = 1/(2*len(y))*np.dot(np.ones(len(y)),(np.dot(X,theta)- y)*(np.dot(X,theta)- y))
    return J

def Grad(X,y,theta,iter,alpha):
    """
    Compute theta with is converge
    theta = theta - alpha/m * X'*(X * theta - y)
    Note: Simultaneous update
    """
    J_store = []
    temp = []
    row, col = X.shape
    for i in range(iter):
        for i in range(col):
            temp.append(theta[i] - (alpha/row) * np.dot(X[:,i].T, np.dot(X,theta)-y))
        theta = np.array(temp).T
        temp = []
        J_store.append(computeJ(X,y,theta))
    return J_store

if __name__ == "__main__":
    plt.plot(Grad(X,y,theta,iter,alpha))
    plt.show()


