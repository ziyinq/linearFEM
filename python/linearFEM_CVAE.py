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

import matplotlib.pyplot as plt
from pyDOE import lhs
from mymodel import CVAE

trainpath = '../data/linear/training'
testpath = '../data/linear/test'

np.random.seed(1234)
    
if __name__ == "__main__":
    
    X = []
    Y = []
    onlyfiles = [f for f in listdir(trainpath) if isfile(join(trainpath, f))]
    for name in onlyfiles:
        Youngs = float(name[15:-4])
        # skip boundary fixed points
        data = loadtxt(trainpath+name, skiprows = 7)
        X.append(Youngs)
        Y.append(data.flatten())
        
    X = np.asarray(X)
    X = X.reshape(X.shape[0],1)
    Y = np.asarray(Y)
    
    X_dim = 1
    Y_dim = Y.shape[1]
    Z_dim = 10
    h_dim = 128
    
    # Model creation
    layers_P = [X_dim + Z_dim , h_dim, h_dim, Y_dim]
    layers_Q = [X_dim + Y_dim, h_dim, h_dim, Z_dim]
    layers_R = [X_dim, h_dim, h_dim, Z_dim]
    
    model = CVAE(X, Y, layers_P, layers_Q, layers_R) 
    
    # Training
    model.train(nIter = 5000, batch_size = Y.shape[0])
      
    testnum = 7
    truedata = Y[testnum]
    testdata = X[testnum]
    testdata = np.reshape(testdata,(1,1))
    samples = model.generate_samples(testdata,1)
    
    test = samples[0]
    plt.figure(1)
    plt.scatter(test[0,0::2], test[0,1::2], c='b')
    plt.scatter(truedata[0::2], truedata[1::2], c='r', marker='x')
    plt.legend(('CVAE data', 'Ground Truth'))
    plt.show()