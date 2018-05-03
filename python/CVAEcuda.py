#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:42:01 2018

@author: ziyin
"""

import torch
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import timeit

class CVAE(torch.nn.Module):
    def __init__(self, X, Y, layers_P, layers_Q, layers_R):
        super(CVAE, self).__init__()
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
#        self.dtype = torch.FloatTensor
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        X = (X - self.Xmean) / self.Xstd
        Y = (Y - self.Ymean) / self.Ystd
            
        # Define PyTorch variables
        X = torch.from_numpy(X).type(self.dtype)
        Y = torch.from_numpy(Y).type(self.dtype)
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)

        self.layers_P = layers_P
        self.layers_Q = layers_Q
        self.layers_R = layers_R
               
        # initialize network 
        self.net_P = self.init_NN(layers_P)
        self.net_Q = self.init_NN(layers_Q)
        self.net_R = self.init_NN(layers_R)
#        linear0 = torch.nn.Linear(self.layers_R[-2], self.layers_R[-1])
#        torch.nn.init.xavier_uniform(linear0.weight)
        self.mu_0 = torch.nn.Linear(self.layers_R[-2], self.layers_R[-1])
        self.Sigma_0 = torch.nn.Linear(self.layers_R[-2], self.layers_R[-1])
        
#        linear1 = torch.nn.Linear(self.layers_Q[-2], self.layers_Q[-1])
#        torch.nn.init.xavier_uniform(linear1.weight)
        self.mu_1 = torch.nn.Linear(self.layers_Q[-2], self.layers_Q[-1])
        self.Sigma_1 = torch.nn.Linear(self.layers_Q[-2], self.layers_Q[-1])
        
#        linear2 = torch.nn.Linear(self.layers_P[-2], self.layers_P[-1])
#        torch.nn.init.xavier_uniform(linear2.weight)
        self.mu_2 = torch.nn.Linear(self.layers_P[-2], self.layers_P[-1])
        self.Sigma_2 = torch.nn.Linear(self.layers_P[-2], self.layers_P[-1])
        
        self.X_dim = X.shape[1]
        self.Y_dim = Y.shape[1]
        self.Z_dim = layers_Q[-1]
    
        # Define optimizer
        # self.optimizer = Adam(self.num_params, lr = 1e-3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        
    def init_NN(self, Q):
        layers = []
        num_layers = len(Q)
        for i in range(0, num_layers-2):
            linear = torch.nn.Linear(Q[i], Q[i+1])
#            torch.nn.init.xavier_uniform(linear.weight)
            layers.append(linear)
            layers.append(torch.nn.Tanh())
        net = torch.nn.Sequential(*layers)
        return net
        
    
    def forward_pass(self):
        X = self.X_batch     
        Y = self.Y_batch     
            
        # Prior: p(z|x)
        temp0 = self.net_R(X)
        mu_0 = self.mu_0(temp0)
        Sigma_0 = self.Sigma_0(temp0)
        
        # Encoder: q(z|x,y)
        temp1 = self.net_Q(torch.cat((X,Y),1))
        mu_1 = self.mu_1(temp1)
        Sigma_1 = self.Sigma_1(temp1)
        
        # Reparametrization trick
        epsilon = Variable(torch.randn(X.shape[0], self.Z_dim))
        Z = mu_1 + torch.exp(Sigma_1 / 2) * epsilon
        
        # Decoder: p(y|x,z)
        temp2 = self.net_P(torch.cat((X, Z), 1))
        mu_2 = self.mu_2(temp2)
        Sigma_2 = self.Sigma_2(temp2)
        
        return mu_0, Sigma_0, mu_1, Sigma_1, mu_2, Sigma_2


    # Fetches a mini-batch of data
    def fetch_minibatch(self,X,Y, N_batch):
        idx = torch.randperm(N_batch)
        X_batch = X[idx,:]
        Y_batch = Y[idx,:]
        return X_batch, Y_batch
    
    
    # Trains the model
    def train(self, nIter = 10000, batch_size = 100):   

        ELBO = np.array([])
        iteration = np.array([])
        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            mu_0, Sigma_0, mu_1, Sigma_1, mu_2, Sigma_2 = self.forward_pass()
            
            # KL[q(z|x,y) || p(z|x)]
            KL = 0.5*(torch.sum(Sigma_1.exp()/Sigma_0.exp()) + torch.sum((mu_0-mu_1)**2/Sigma_0.exp()) - self.Z_dim + torch.sum(Sigma_0) - torch.sum(Sigma_1))
            
            # -log p(y|x,z)
            NLML = 0.5*(torch.sum((self.Y_batch-mu_2)**2/Sigma_2.exp()) + torch.sum(Sigma_2) + np.log(2.*np.pi)*self.Y_dim*self.X_batch.shape[0])

            loss = KL + NLML
            
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            
            # Reset gradients for next step
            self.optimizer.zero_grad()
            
            # Print
            if it % 50 == 0:
                elapsed = timeit.default_timer() - start_time
                iteration = np.append(iteration, it)
                ELBO = np.append(ELBO, loss)
                print('It: %d, ELBO: %.3e, Time: %.2f' % 
                      (it, loss, elapsed))
                start_time = timeit.default_timer()
                
        return ELBO, iteration
        
      
    def generate_samples(self, X_star, N_samples):
        X_star = (X_star - self.Xmean) / self.Xstd
        X_star = torch.from_numpy(X_star).type(self.dtype)
        X_star = Variable(X_star, requires_grad=False)
        # Encode X_star
        temp0 = self.net_R(X_star)
        mu_0 = self.mu_0(temp0)
        Sigma_0 = self.Sigma_0(temp0)
        
        # Reparametrization trick
        epsilon = Variable(torch.randn(N_samples, self.Z_dim))
        Z = mu_0 + torch.exp(Sigma_0 / 2) * epsilon
                
        # Decode
        temp2 = self.net_P(torch.cat((X_star, Z), 1))
        mean_star = self.mu_2(temp2)
        var_star = self.Sigma_2(temp2)
        
        mean_star = mean_star.data.numpy()
        var_star = var_star.data.numpy()
        # De-normalize
        mean_star = mean_star*self.Ystd + self.Ymean
        var_star = var_star*self.Ystd**2
        
        return mean_star, var_star
    