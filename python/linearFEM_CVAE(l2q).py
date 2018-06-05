#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:12:39 2018

@author: ziyin
"""
from os import listdir
from os.path import isfile, join
from numpy import loadtxt
import numpy as np

from CVAEcuda import CVAE
import matplotlib.pyplot as plt
from pyDOE import lhs
#from mymodel import CVAE

trainpathL = '../data/l2q/train/'
trainpathQ = '../data/l2q/train/quad/'
testpathL = '../data/l2q/test/'
testpathQ = '../data/l2q/test/quad/'

np.random.seed(1234)
    
if __name__ == "__main__":
    
    # Get trainning data
    X = []
    Y = []
    onlyfiles = [f for f in listdir(trainpathL) if isfile(join(trainpathL, f))]
    c = 0
    for name in onlyfiles:
        Youngs = float(name[15:-4])
        # skip boundary fixed points
        data = loadtxt(trainpathL+name, skiprows = 7)
        data = data.flatten().tolist()
        X.append(data)
        X[c].append(Youngs)
        c = c + 1
        
    onlyfiles = [f for f in listdir(trainpathQ) if isfile(join(trainpathQ, f))]
    for name in onlyfiles:
        data = loadtxt(trainpathQ+name, skiprows = 7)
        Y.append(data.flatten())
        
    # Get test data
    testX = []
    testY = []
    onlyfiles = [f for f in listdir(testpathL) if isfile(join(testpathL, f))]
    c = 0
    for name in onlyfiles:
        Youngs = float(name[15:-4])
        # skip boundary fixed points
        data = loadtxt(testpathL+name, skiprows = 7)
        data = data.flatten().tolist()
        testX.append(data)
        testX[c].append(Youngs)
        c = c + 1
    
    onlyfiles = [f for f in listdir(testpathQ) if isfile(join(testpathQ, f))]
    for name in onlyfiles:
        data = loadtxt(testpathQ+name, skiprows = 7)
        testY.append(data.flatten())
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    testX = np.asarray(testX)
    testY = np.asarray(testY)
    
    X_dim = X.shape[1]
    Y_dim = Y.shape[1]
    Z_dim = 10
    h_dim = 128
    
    # Model creation
    layers_P = [X_dim + Z_dim , h_dim, h_dim, Y_dim]
    layers_Q = [X_dim + Y_dim, h_dim, h_dim, Z_dim]
    layers_R = [X_dim, h_dim, h_dim, Z_dim]
    
    model = CVAE(X, Y, layers_P, layers_Q, layers_R) 
    
    # Training
    ELBO, iteration = model.train(nIter = 40000, batch_size = Y.shape[0])
      
    # plot
    testnum = 1
    truedata = testY[testnum]
    inputdata = testX[testnum]
    inputdata = np.reshape(inputdata,(X_dim,1))
    CVAE_data = model.generate_samples(inputdata,1)
    
    CVAE_data = CVAE_data[0]
    plt.figure(1)
    plt.scatter(CVAE_data[0,0::2], CVAE_data[0,1::2], c='b')
    plt.scatter(truedata[0::2], truedata[1::2], c='r', marker='x')
    plt.legend(('Prediction', 'Ground Truth'))
    plt.title('Youngs Modulus: ' + str(inputdata[-1,0]))
    plt.show()
    
    plt.figure(2)
    plt.plot(iteration, ELBO)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('Convergence Plot')
    plt.show()