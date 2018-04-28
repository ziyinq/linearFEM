#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:52:41 2018

@author: Paris
"""

import autograd.numpy as np
from autograd import grad
from optimizers import Adam
import timeit

class VAE:
    def __init__(self, X, layers_Q, layers_P):
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        X = (X - self.Xmean) / self.Xstd
            
        self.X = X

        self.layers_Q = layers_Q
        self.layers_P = layers_P
        
        self.X_dim = X.shape[1]
        self.Z_dim = layers_Q[-1]
    
        # Initialize encoder
        params =  self.initialize_NN(layers_Q)
        self.idx_Q = np.arange(params.shape[0])
        
        # Initialize decoder
        params = np.concatenate([params, self.initialize_NN(layers_P)])
        self.idx_P = np.arange(self.idx_Q[-1]+1, params.shape[0])
                
        self.params = params
        
       # Total number of parameters
        self.num_params = self.params.shape[0]
        
        # Define optimizer
        self.optimizer = Adam(self.num_params, lr = 1e-3)
        
        # Define gradient function using autograd 
        self.grad_elbo = grad(self.ELBO)
        
        
    def initialize_NN(self, Q):
        hyp = np.array([])
        layers = len(Q)
        for layer in range(0,layers-2):
            A = -np.sqrt(6.0/(Q[layer]+Q[layer+1])) + 2.0*np.sqrt(6.0/(Q[layer]+Q[layer+1]))*np.random.rand(Q[layer],Q[layer+1])
            b = np.zeros((1,Q[layer+1]))
            hyp = np.concatenate([hyp, A.ravel(), b.ravel()])

        A = -np.sqrt(6.0/(Q[-2]+Q[-1])) + 2.0*np.sqrt(6.0/(Q[-2]+Q[-1]))*np.random.rand(Q[-2],Q[-1])
        b = np.zeros((1,Q[-1]))
        hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        A = -np.sqrt(6.0/(Q[-2]+Q[-1])) + 2.0*np.sqrt(6.0/(Q[-2]+Q[-1]))*np.random.rand(Q[-2],Q[-1])
        b = np.zeros((1,Q[-1]))
        hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        return hyp
        
    
    def forward_pass(self, X, Q, params):
        H = X
        idx_3 = 0
        layers = len(Q) 
        for layer in range(0,layers-2):        
            idx_1 = idx_3
            idx_2 = idx_1 + Q[layer]*Q[layer+1]
            idx_3 = idx_2 + Q[layer+1]
            A = np.reshape(params[idx_1:idx_2], (Q[layer],Q[layer+1]))
            b = np.reshape(params[idx_2:idx_3], (1,Q[layer+1]))
            H = np.tanh(np.matmul(H,A) + b)
            
        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        A = np.reshape(params[idx_1:idx_2], (Q[-2],Q[-1]))
        b = np.reshape(params[idx_2:idx_3], (1,Q[-1]))
        mu = np.matmul(H,A) + b

        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        A = np.reshape(params[idx_1:idx_2], (Q[-2],Q[-1]))
        b = np.reshape(params[idx_2:idx_3], (1,Q[-1]))
        Sigma = np.exp(np.matmul(H,A) + b)
        
        return mu, Sigma
    
    
    def ELBO(self, params):
        X = self.X_batch     
            
        # Prior: p(z)
        epsilon = np.random.randn(X.shape[0],self.Z_dim) 
        
        # Encoder: q(z|x)
        mu_1, Sigma_1 = self.forward_pass(X, 
                                          self.layers_Q, 
                                          params[self.idx_Q]) 
        
        # Reparametrization trick
        Z = mu_1 + epsilon*np.sqrt(Sigma_1)
        
        # Decoder: p(x|z)
        mu_2, Sigma_2 = self.forward_pass(Z, 
                                          self.layers_P, 
                                          params[self.idx_P])
        
        # Log-determinants
        log_det_1 = np.sum(np.log(Sigma_1))
        log_det_2 = np.sum(np.log(Sigma_2))
        
        # KL[q(z|y) || p(z)]
        KL = 0.5*(np.sum(Sigma_1) + np.sum(mu_1**2) - self.Z_dim - log_det_1)
        
        # -log p(y)
        NLML = 0.5*(np.sum((X-mu_2)**2/Sigma_2) + log_det_2 + np.log(2.0*np.pi)*self.X_dim*X.shape[0])
           
        return NLML + KL
    

    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        return X_batch
    
    
    # Trains the model
    def train(self, nIter = 10000, batch_size = 100):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch = self.fetch_minibatch(self.X, batch_size)
            
            # Evaluate loss using current parameters
            params = self.params
            elbo = self.ELBO(params)
          
            # Update parameters
            grad_params = self.grad_elbo(params)
            self.params = self.optimizer.step(params, grad_params)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, ELBO: %.3e, Time: %.2f' % 
                      (it, elbo, elapsed))
                start_time = timeit.default_timer()
        
        
    def generate_samples(self, N_samples):          
        # Prior: p(z)
        Z = np.random.randn(N_samples, self.Z_dim)        
        
        # Decoder: p(x|z)
        mean_star, var_star = self.forward_pass(Z, 
                                                self.layers_P, 
                                                self.params[self.idx_P]) 
                    
        return mean_star, var_star
    
    
    def predict_Z(self, X_star):
        X_star = (X_star - self.Xmean) / self.Xstd
        # Encoder: q(z|x)
        mean_star, var_star = self.forward_pass(X_star, 
                                                self.layers_Q, 
                                                self.params[self.idx_Q]) 
                    
        return mean_star, var_star
        

class CVAE:
    def __init__(self, X, Y, layers_P, layers_Q, layers_R):
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        X = (X - self.Xmean) / self.Xstd
        Y = (Y - self.Ymean) / self.Ystd
            
        self.X = X
        self.Y = Y

        self.layers_P = layers_P
        self.layers_Q = layers_Q
        self.layers_R = layers_R
               
        self.X_dim = X.shape[1]
        self.Y_dim = Y.shape[1]
        self.Z_dim = layers_Q[-1]
    
        # Initialize encoder
        params =  self.initialize_NN(layers_P)
        self.idx_P = np.arange(params.shape[0])
        
        # Initialize decoder
        params = np.concatenate([params, self.initialize_NN(layers_Q)])
        self.idx_Q = np.arange(self.idx_P[-1]+1, params.shape[0])
        
        # Initialize prior
        params = np.concatenate([params, self.initialize_NN(layers_R)])
        self.idx_R = np.arange(self.idx_Q[-1]+1, params.shape[0])
                
        self.params = params
        
       # Total number of parameters
        self.num_params = self.params.shape[0]
        
        # Define optimizer
        self.optimizer = Adam(self.num_params, lr = 1e-3)
        
        # Define gradient function using autograd 
        self.grad_elbo = grad(self.ELBO)
        
        
    def initialize_NN(self, Q):
        hyp = np.array([])
        layers = len(Q)
        for layer in range(0,layers-2):
            A = -np.sqrt(6.0/(Q[layer]+Q[layer+1])) + 2.0*np.sqrt(6.0/(Q[layer]+Q[layer+1]))*np.random.rand(Q[layer],Q[layer+1])
            b = np.zeros((1,Q[layer+1]))
            hyp = np.concatenate([hyp, A.ravel(), b.ravel()])

        A = -np.sqrt(6.0/(Q[-2]+Q[-1])) + 2.0*np.sqrt(6.0/(Q[-2]+Q[-1]))*np.random.rand(Q[-2],Q[-1])
        b = np.zeros((1,Q[-1]))
        hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        A = -np.sqrt(6.0/(Q[-2]+Q[-1])) + 2.0*np.sqrt(6.0/(Q[-2]+Q[-1]))*np.random.rand(Q[-2],Q[-1])
        b = np.zeros((1,Q[-1]))
        hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        return hyp
        
    
    def forward_pass(self, X, Q, params):
        H = X
        idx_3 = 0
        layers = len(Q)
        for layer in range(0,layers-2):        
            idx_1 = idx_3
            idx_2 = idx_1 + Q[layer]*Q[layer+1]
            idx_3 = idx_2 + Q[layer+1]
            A = np.reshape(params[idx_1:idx_2], (Q[layer],Q[layer+1]))
            b = np.reshape(params[idx_2:idx_3], (1,Q[layer+1]))
            H = np.tanh(np.matmul(H,A) + b)
            
        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        A = np.reshape(params[idx_1:idx_2], (Q[-2],Q[-1]))
        b = np.reshape(params[idx_2:idx_3], (1,Q[-1]))
        mu = np.matmul(H,A) + b

        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        A = np.reshape(params[idx_1:idx_2], (Q[-2],Q[-1]))
        b = np.reshape(params[idx_2:idx_3], (1,Q[-1]))
        Sigma = np.exp(np.matmul(H,A) + b)
        
        return mu, Sigma
    
    
    def ELBO(self, params):
        X = self.X_batch     
        Y = self.Y_batch     
            
        # Prior: p(z|x)
        mu_0, Sigma_0 = self.forward_pass(X, 
                                          self.layers_R, 
                                          params[self.idx_R]) 
        
        # Encoder: q(z|x,y)
        mu_1, Sigma_1 = self.forward_pass(np.concatenate([X, Y], axis = 1), 
                                          self.layers_Q, 
                                          params[self.idx_Q]) 
        
        # Reparametrization trick
        epsilon = np.random.randn(X.shape[0],self.Z_dim)       
        Z = mu_1 + epsilon*np.sqrt(Sigma_1)
        
        # Decoder: p(y|x,z)
        mu_2, Sigma_2 = self.forward_pass(np.concatenate([X, Z], axis = 1), 
                                          self.layers_P, 
                                          params[self.idx_P])
        
        # Log-determinants
        log_det_0 = np.sum(np.log(Sigma_0))
        log_det_1 = np.sum(np.log(Sigma_1))
        log_det_2 = np.sum(np.log(Sigma_2))
        
        # KL[q(z|x,y) || p(z|x)]
        KL = 0.5*(np.sum(Sigma_1/Sigma_0) + np.sum((mu_0-mu_1)**2/Sigma_0) - self.Z_dim + log_det_0 - log_det_1)
        
        # -log p(y|x,z)
        NLML = 0.5*(np.sum((Y-mu_2)**2/Sigma_2) + log_det_2 + np.log(2.*np.pi)*self.Y_dim*X.shape[0])
        
        return NLML + KL
    

    # Fetches a mini-batch of data
    def fetch_minibatch(self,X,Y, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        Y_batch = Y[idx,:]
        return X_batch, Y_batch
    
    
    # Trains the model
    def train(self, nIter = 10000, batch_size = 100):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Evaluate loss using current parameters
            params = self.params
            elbo = self.ELBO(params)
          
            # Update parameters
            grad_params = self.grad_elbo(params)
            self.params = self.optimizer.step(params, grad_params)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, ELBO: %.3e, Time: %.2f' % 
                      (it, elbo, elapsed))
                start_time = timeit.default_timer()
        
      
    def generate_samples(self, X_star, N_samples):
        X_star = (X_star - self.Xmean) / self.Xstd
        # Encode X_star
        mu_0, Sigma_0 = self.forward_pass(X_star, 
                                          self.layers_R, 
                                          self.params[self.idx_R]) 
        
        # Reparametrization trick
        epsilon = np.random.randn(N_samples,self.Z_dim)        
        Z = mu_0 + epsilon*np.sqrt(Sigma_0)
                
        # Decode
        mean_star, var_star = self.forward_pass(np.concatenate([X_star, Z], axis = 1), 
                                                self.layers_P, 
                                                self.params[self.idx_P]) 
             
        # De-normalize
        mean_star = mean_star*self.Ystd + self.Ymean
        var_star = var_star*self.Ystd**2
        
        return mean_star, var_star
    
    
    
