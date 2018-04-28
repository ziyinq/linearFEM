#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:41:17 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from models import CVAE

np.random.seed(1234)
    
if __name__ == "__main__":
    
    N = 200
    X_dim = 1
    Y_dim = 1
    Z_dim = 5
    noise = 0.05
    
    
    lb = 0.0*np.ones((1,X_dim))
    ub = 2.0*np.ones((1,X_dim)) 
    
    # Generate training data
    def f(x):
        return x*np.sin(np.pi*x)

    # Data
    X = lb + (ub-lb)*lhs(X_dim, N)
    Y = f(X) + noise*np.random.randn(N,Y_dim)
    
    
    # Generate test data
    N_star = 400
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    Y_star = f(X_star)
    
            
    # Model creation
    layers_P = [X_dim + Z_dim , 50, 50, Y_dim]
    layers_Q = [X_dim + Y_dim, 50, 50, Z_dim]
    layers_R = [X_dim, 50, 50, Z_dim]
    
    model = CVAE(X, Y, layers_P, layers_Q, layers_R) 
    
    # Training
    model.train(nIter = 20000, batch_size = Y.shape[0])
      
    # Prediction
    plt.figure(1, facecolor = 'w')
    N_samples = 1000
    Y_samples = np.zeros((X_star.shape[0], Y.shape[1], N_samples))
    for i in range(0, N_samples):
        Y_samples[:,:,i], _ = model.generate_samples(X_star, 1)
#        plt.plot(X_star, Y_samples[:,:,i], 'ro', alpha = 0.1)   
#    plt.plot(X_star, Y_star, 'b-', label = "Exact", linewidth=2)
#    plt.plot(X,Y,'bo',alpha = 0.8)
#    plt.xlabel('$x$')
#    plt.ylabel('$f(x)$')
    
    Y_pred = np.mean(Y_samples, axis = 2)    
    Y_var = np.var(Y_samples, axis = 2)
        
    
    plt.plot(X_star, Y_pred, 'r--', label = "Prediction", linewidth=2)   
    lower = Y_pred - 2.0*np.sqrt(Y_var)
    upper = Y_pred + 2.0*np.sqrt(Y_var)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    plt.plot(X,Y,'bo',alpha = 0.5)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
#
