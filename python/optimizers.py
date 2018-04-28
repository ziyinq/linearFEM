#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:26:35 2018

@author: Paris
"""

import numpy as np

class SGD:
    def __init__(self, num_params, lr=1e-3, momentum = 0.0):
        self.lr = lr 
        self.momentum = momentum
        self.velocity = np.zeros(num_params)

    def step(self, w, grad_w):
        self.velocity = self.momentum * self.velocity - (1.0 - self.momentum) * grad_w
        w = w + self.lr * self.velocity
        return w
    
    
class Adam:
    def __init__(self, num_params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):        
        self.lr = lr                
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.it_cnt = 0        
        self.mt = np.zeros(num_params)     
        self.vt = np.zeros(num_params)  
        
    def step(self, w, grad_w):        
        self.it_cnt = self.it_cnt + 1        
        self.mt = self.mt*self.beta1 + (1.0-self.beta1)*grad_w
        self.vt = self.vt*self.beta2 + (1.0-self.beta2)*grad_w**2
        mt_hat = self.mt/(1.0-self.beta1**self.it_cnt)
        vt_hat = self.vt/(1.0-self.beta2**self.it_cnt)
        w = w - (self.lr/(np.sqrt(vt_hat) + self.epsilon))*mt_hat            
        return w
        

class RMSprop:
    def __init__(self, num_params, lr=1e-3, gamma=0.9, epsilon=1e-8):
        self.lr = lr 
        self.gamma = gamma
        self.epsilon = epsilon
        self.avg_sq_grad = np.ones(num_params)

    def step(self,w,grad_w):        
        self.avg_sq_grad = self.avg_sq_grad * self.gamma + grad_w**2 * (1.0 - self.gamma)
        w = w - self.lr * grad_w/(np.sqrt(self.avg_sq_grad) + self.epsilon)
        return w    
