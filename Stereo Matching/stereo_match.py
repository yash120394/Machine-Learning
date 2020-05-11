# -*- coding: utf-8 -*-
"""
Created on Mon May 11 02:21:03 2020

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

from scipy import stats
import matplotlib.pyplot as plt
import math
from copy import deepcopy


# Load left and right pictures

Xl = plt.imread('im0.ppm', format = None)
Xr = plt.imread('im8.ppm', format = None)

print('Shape of left picture :',Xl.shape)
print('Shape of left picture :',Xr.shape)


# plot two pictures: left and right

fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,10)
ax[0].imshow(Xl)
ax[1].imshow(Xr)


# Calculate the disparity matrix 

D = np.zeros((Xl[:,:,0].shape[0],Xl[:,:,0].shape[1]-40)) 

def euclidean_dist(X,Y):
    return np.sqrt(np.sum((X-Y)**2))

for i in range(Xr.shape[0]):
    for j in range(Xr.shape[1]-40):
        dist = [euclidean_dist(Xl[i,j+k],Xr[i,j]) for k in range(40)]
        ind_dist = np.argmin(dist)
        D[i,j] = ind_dist
        
        
# Vectorize disparity matrix and draw a histogram

D_flatten = D.flatten()
plt.hist(D_flatten, bins = 40)
plt.xlabel('Disparity')
plt.ylabel('Frequency')      


# random initialisation of cluster value for samples

np.random.seed(902)
K = 3
df_D = pd.DataFrame(D_flatten)
df_D['cluster'] = np.random.choice([0,1,2],len(df_D))
df_D.columns = ['X','cluster']


# calculating mean vector of individual clusters

mu=[np.mean(df_D[df_D['cluster'] == i])['X'] for i in range(K)]
mu = np.array(mu)
mu = np.reshape(mu,(K,1))

# calculating standard deviation of individual clusters

sigma = [np.std(df_D[df_D['cluster'] == i])['X'] for i in range(K)]
sigma = np.array(sigma)
sigma = np.reshape(sigma, (K,1))

# calculating prior probabilities of clusters

prior = [len(df_D[df_D['cluster'] == i])/len(df_D) for i in range(K)]
prior = np.array(prior)
prior = np.reshape(prior, (K,1))


# define function for calculating pdf

def calculate_pdf(X,mean,sd):
    var = float(sd)**2
    den = (2*math.pi*var)**0.5
    num = math.exp(-(float(X)-float(mean))**2/(2*var))
    return num/den

# initialise U matrix for E step calculation

U = np.zeros((len(df_D),K))

# GMM algorithm

for iter in range(50):
    # Updating U matrix : E-step
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            numerator = prior[j]*calculate_pdf(df_D.iloc[i,0],mu[j],sigma[j])
            denominator = prior[0]*calculate_pdf(df_D.iloc[i,0],mu[0],sigma[0]) + prior[1]*calculate_pdf(df_D.iloc[i,0],mu[1],sigma[1]) + prior[2]*calculate_pdf(df_D.iloc[i,0],mu[2],sigma[2])
            U[i,j] = numerator/denominator        
    
    mu_old = mu.copy()
    sigma_old = sigma.copy()
    
    # Updating mu, prior and sigma : M-step
    for k in range(K):
        mu[k] = np.dot(U[:,k].transpose(),df_D['X'])/(np.sum(U[:,k]))
    for k in range(K):
        prior[k] = np.sum(U[:,k])/len(df_D)
    for k in range(K):
        sigma[k] = (np.sum(np.multiply(U[:,k],(df_D['X'] - mu[k])**2)) / np.sum(U[:,k]))**0.5
    
    print("Iteration no:",iter)
    print("Mean cluster vector:", mu)
    print("\n")


# Assign new clusters based on posterior probabilities

df_D['new_cluster'] = 0
for i in range(len(df_D['X'])):
    post_cluster = []
    for j in range(K):
        posterior = calculate_pdf(df_D.iloc[i,0],mu[j],sigma[j])
        post_cluster.append(posterior)
    df_D.iloc[i,2] = np.argmax(post_cluster)


# plot disparity distribution of different clusters (K = 3) 

unique_clusters = df_D.new_cluster.unique()
plt.hist([df_D.loc[df_D.new_cluster == x, 'X'] for x in range(K)], label=unique_clusters, bins = 20)
plt.legend(['Cluster0', 'Cluster1', 'Cluster2'])

plt.title('Disparity distribution for different clusters')
plt.xlabel('Disparity')
plt.ylabel('Frequency')


# Replace disparity values with cluster means

df_D['mean_disparity'] = 0
for i in range(len(df_D)):
    for j in range(K):
        if df_D.iloc[i,2] == j :
            df_D.iloc[i,3] = mu[j]


# Create a new disparity map of K level

D_flatten_new = np.array(df_D['mean_disparity'])
D_new = np.reshape(D_flatten_new, (381,390))

# Plotting the new disparity matrix of K level and comparing the result with left and right camera picture

fig, ax = plt.subplots(1,3)
fig.set_size_inches(10,10)
ax[0].imshow(Xl,cmap = "gray")
ax[1].imshow(Xr,cmap = "gray")
ax[2].imshow(D_new,cmap = "gray")


# Initialising depth map and cluster array

disparity_map_list = []
cluster_array = np.array(df_D['new_cluster'])
cluster_array = np.reshape(cluster_array,(381,390))
cluster_array_old = np.array(df_D['new_cluster'])
cluster_array_old = np.reshape(cluster_array,(381,390))
disparity_map_smooth = np.zeros(cluster_array.shape)


# define a similarity function using exponential function

def similarity_func(i,j,cluster):
    val = 5
    var = 0.5
    if cluster_array_old[i,j] == cluster:
        val = 0
    return np.exp(-(val*val/(2*var*var)))

# defining function for neighbouring probability 

def smoothing_prob(i,j,cluster):
    N = [-1,0,1]
    smooth_p = 1
    for k in N:
        for l in N:
            smooth_p *= similarity_func(i+k,j+l, cluster)
    return smooth_p

# Gibbs sampling algorithm

for iteration in range(30):
    print(iteration)
    print("\n")
    for i in range(1, disparity_map_smooth.shape[0]-1):
        for j in range(1, disparity_map_smooth.shape[1]-1):
            current_cluster = cluster_array[i,j]
            posterior = np.zeros(K)
            for k in range(K):
                posterior[k] = calculate_pdf(D_new[i,j],mu[k],sigma[k]) * smoothing_prob(i,j,k)
            posterior = posterior/np.sum(posterior)
            new_label = np.random.choice(np.arange(0, K), p=posterior)
            disparity_map_smooth[i,j] = mu[new_label]
            cluster_array[i,j] = new_label
    disparity_map_list.append(disparity_map_smooth)
    cluster_array_old = np.array(cluster_array)


# Plot new smooth disparity map
# majority voting using last 10 samples generated

disparity_map_list_array = np.array(disparity_map_list)
disparity_map_smooth1 = scipy.stats.mode(disparity_map_list_array[-10:])[0][0]

fig, ax = plt.subplots(1,3)
fig.set_size_inches(10,10)
ax[0].imshow(Xl,cmap = "gray")
ax[1].imshow(Xr,cmap = "gray")
ax[2].imshow(disparity_map_smooth1,cmap="gray")





  