# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:23:53 2020

@author: yash1
"""

# Load required libraries

import pandas as pd
import numpy as np
import scipy as sp
from scipy import io
import matplotlib.pyplot as plt
import math
import itertools
import random
from scipy import linalg
from scipy import stats
from copy import deepcopy

# load trx and try matrix 

X = io.loadmat('trX.mat')
Y = io.loadmat('trY.mat')

X = X['trX']
Y = Y['trY']

print("Shape of input X matrix:", X.shape)
print("Shape of input Y matrix:", Y.shape)


# train a weak classifier function

np.random.seed(54)
def train_perceptron(x, y, weights):
    thetas = np.random.randn(2,1)*np.sqrt(2/1)
    bias = np.random.rand(1,1)*np.sqrt(2/1)
    lr = 0.005
    J = []
    for i in range(2000):
        if i > 1200:
            lr = 0.0001
        xw = np.dot(thetas.transpose(), x) + bias
        yhat = np.tanh(xw)
        error = (1/2)*weights*((yhat - y)**2)
        error_sum = np.sum(error)
        J.append(error_sum)
        grad = (yhat-y)*weights*(1 - np.tanh(xw)**2)
        thetas -= lr*np.dot(x, grad.transpose())
        bias -= np.sum(lr*grad)
    return np.sign(yhat), thetas, bias


# Train a ada boost classifier with weak perceptron classifier

def AdaBoostClassifier(x, y, n_class):
    w = (1/y.shape[1])*np.ones(y.shape)
    w_list = []
    phi_list = []
    beta_list = []
    t_list = []
    b_list = []
    for c in range(n):
        phi, t, b = train_perceptron(x, y, w)
        phi_list.append(phi)
        t_list.append(t)
        b_list.append(b)
        right = 0 
        wrong = 0
        for i in range(y.shape[1]):
            if phi[0,i] == y[0,i]:
                right += w[0,i]
            else :
                wrong += w[0,i]
                
        beta = (1/2)*np.log(right/wrong)
        beta_list.append(beta)
        
        w = w*np.exp(-beta*y*phi)
        w = w/np.sum(w)
        w_list.append(w)
        
    
    
    phi_array = np.array(phi_list)
    phi_array = phi_array.reshape(n,y.shape[1])
    beta_array = np.array(beta_list)
    beta_array = beta_array.reshape(n,1)
    t_array = np.array(t_list)
    t_array = t_array.reshape(n,x.shape[0])
    b_array = np.array(b_list)
    b_array = b_array.reshape(n,1)
    
    return beta_array, phi_array, w_list, t_array, b_array


# Calculate accuracy for n weak classifiers

n_class = [200, 500, 1000, 1200]
accuracy_list = []

for n in n_class:
    b,phi_x,w_list,theta,bias = AdaBoostClassifier(X, Y, n)
    w_array = np.array(w_list)
    w_array = w_array.reshape(-1,1)
    fm = b*phi_x
    Y_pred = np.sum(fm, axis = 0)
    Y_pred1 = np.sign(Y_pred.reshape(1,Y.shape[1]))
    correct = 0
    for i in range(Y.shape[1]):
        if Y[0,i] == Y_pred1[0,i]:
            correct += 1
    accuracy = (correct/Y.shape[1])*100
    accuracy_list.append(accuracy)
    print("Number of weak classifiers:", n)
    print("Accuracy:", accuracy)
    print("\n")


# Plot the Ada boost classifier results

## getting x and y coordinates of metal music
x_metal = X[0, np.where(Y == 1)[1]]
y_metal = X[1, np.where(Y == 1)[1]]

## getting x and y coordinates of rock music
x_rock = X[0, np.where(Y == -1)[1]]
y_rock = X[1, np.where(Y == -1)[1]]

## Creating a mesh grid from orginal train X matrix 
min_x = np.min(X[0,:]) - 0.05
max_x = np.max(X[0,:]) + 0.05
min_y = np.min(X[1,:]) - 0.05
max_y = np.max(X[1,:]) + 0.05

x1, y1 = np.meshgrid(np.arange(min_x, max_x, 0.005),np.arange(min_y, max_y, 0.005))

XY_flatten = np.c_[x1.flatten(), y1.flatten()].T

# Contour plot to differentiate rock and metal music 

ctr = np.sum(b*(np.tanh(np.dot(theta, XY_flatten) + bias)), axis = 0)

fig, ax = plt.subplots()
contour = ax.contourf(x1, y1, ctr.reshape(x1.shape), 100)
ax.scatter(x_metal, y_metal, s=1000*w_array)
ax.scatter(x_rock, y_rock, s=1000*w_array)
plt.xlabel('loud')
plt.ylabel('noise')
plt.title('Adaboost result on 1200 classifiers')
plt.show()
