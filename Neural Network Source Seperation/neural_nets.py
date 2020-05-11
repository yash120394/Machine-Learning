# -*- coding: utf-8 -*-
"""
Created on Mon May 11 01:13:35 2020

@author: yash1
"""
# Load required libraries

import pandas as pd
import numpy as np
import scipy as sp
import scipy.io.wavfile
import matplotlib.pyplot as plt
import math
from IPython.display import Audio
import itertools
import random
import matplotlib.image as img
from scipy import linalg
from scipy.signal import stft 
from scipy.signal import istft
from scipy import stats
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import librosa.output


# Load speech and noise signals

fs, s = sp.io.wavfile.read("trs.wav")
fn, n = sp.io.wavfile.read("trn.wav")
print("Speech shape:", s.shape)
print("Sampling rate of speech:", fs)
print("Noise shape:", n.shape)
print("Sampling rate of noise:", fn)

# Add them to get noisy signal

x = s + n
print("Noisy signal shape:", x.shape)


# Apply STFT to signal, noise and mixture

S = stft(s, fs=16000, window='hann', nperseg=1024)[2]
N = stft(n, fs=16000, window='hann', nperseg=1024)[2]
X = stft(x, fs=16000, window='hann', nperseg=1024)[2]

print("Spectogram shape of speech:", S.shape)
print("Spectogram shape of noise:", N.shape)
print("Spectogram shape of mixture:", X.shape)


# Define a Ideal Binary Mask (IBM)

M = np.zeros(S.shape)
for i in range(len(M)):
    for j in range(len(M[0])):
        if np.abs(S[i,j]) > np.abs(N[i,j]):
            M[i,j] = 1
        else:
            M[i,j] = 0
            

# Define activation function 

def sigmoid(z):
    return 1/(1+np.exp(-z))

def Relu(z):
    return z * (z > 0)

def dRelu(z):
    return 1. * (z > 0)



# initialising weights for hidden layer and final layer

np.random.seed(3)
weight1 = np.random.randn(513,50)*np.sqrt(2/50) 
bias1 = np.random.randn(50,1)*np.sqrt(2)
weight2 = np.random.randn(50,513)*np.sqrt(2/513) 
bias2 = np.random.randn(513,1)*np.sqrt(2) 
alpha = 0.00001


# Define input and output matrix

X1 = np.absolute(X)
Y = M


# Neural Network 

J = []
alpha1 = alpha
for i in range(10000):
    if i > 5000:
        alpha1 = 0.00005
    else : 
        pass
    # forward propogation
    Z1 = np.dot(weight1.transpose(), X1) + bias1
    A1 = Relu(Z1)
    Z2 = np.dot(weight2.transpose(), A1) + bias2
    A2 = sigmoid(Z2)
    
    # calculating error 
    avg_error = np.sum(0.5*(A2-Y)**2)/((Y.shape[0])*(Y.shape[1]))
    J.append(avg_error)
    
    # back propogation
    grad2 = (A2-Y)*sigmoid(Z2)*(1-sigmoid(Z2))
    grad1 = np.dot(weight2,(A2-Y)*sigmoid(Z2)*(1-sigmoid(Z2)))*dRelu(Z1)
    
    #grad1 = weight2@((A2-Y)*sigmoid(Z2)*(1-sigmoid(Z2)))*(1-np.tanh(Z1)**2)

    # update weight1 and weight2
    weight2 -= alpha1*np.dot(A1,grad2.T)
    weight1 -= alpha1*np.dot(X1,grad1.T)
    
    # update bias1 and bias2
    bias2 -= np.sum(alpha1*grad2)
    bias1 -= np.sum(alpha1*grad1)

# Plotting loss function vs number of steps

plt.plot(J)
plt.ylabel('loss');
plt.xlabel('number of steps');
plt.title('Loss function vs steps for NN classifier')            


# load test noisy and ground truth clean speech

fxtest, xtest = sp.io.wavfile.read("tex.wav")
fsclean, sclean = sp.io.wavfile.read("tes.wav")

# scale noisy and clean test speech

xtest = xtest/np.iinfo(np.int16).max
sclean = sclean/np.iinfo(np.int16).max


# Apply STFT to test noisy signal

Xtest = stft(xtest, fs=16000, window='hann', nperseg=1024)[2]


# predict mask on STFT of test noisy signal

Xtest1 = np.absolute(Xtest)

Ztest1 = np.dot(weight1.transpose(), Xtest1) + bias1
Atest1 = Relu(Ztest1)
Ztest2 = np.dot(weight2.transpose(), Atest1) + bias2
Atest2 = sigmoid(Ztest2)

# recover test clean speech spectogram from predicted mask matrix

Mtest = Atest2
Sclean_pred = Xtest*Mtest

# getting inverse stft of predicted clean speech

sclean_pred = istft(Sclean_pred, fs=16000)[1]

# match the length of clean test speech to calculate SNR

sclean_pred1 = sclean_pred[:len(sclean)]

# save the predicted audio file

scipy.io.wavfile.write("tes_pred.wav", 16000, sclean_pred1)

# plot clean speech waveform

plt.plot(sclean)

# plot clean predicted speech waveform

plt.plot(sclean_pred1)


# report signal to noise ratio 

num = sum(sclean**2)
den = sum((sclean-sclean_pred1)**2)

SNR = 10*np.log10(num/den)
print("Signal to Noise ratio of predicted clean speech is:", SNR)
