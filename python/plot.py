import matplotlib.pyplot as plt
from numpy import loadtxt

linearData = loadtxt("../data/linear/origin/linearFrame250E15000.txt")
quadData = loadtxt("../data/quad/quadFrame250E15000.txt")
plt.figure(1)
plt.scatter(linearData[:,0], linearData[:,1], c='b')
plt.scatter(quadData[:,0], quadData[:,1], c='r', marker='x')
plt.legend(('Linear FEM', 'Quadratic FEM'))
plt.show()
