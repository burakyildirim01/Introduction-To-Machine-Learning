import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.spatial as spa
from scipy.sparse.linalg import eigs
import networkx as nx

def update_centroids(memberships, X):
    if memberships is None:
        centroids = X[np.random.choice(range(N), K, False),:]
    else:
        centroids = np.vstack([np.mean(X[memberships == k,], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10, 
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")

G = nx.DiGraph() 

data_set = np.genfromtxt('hw06_data_set.csv',delimiter=',',skip_header=1)
N = len(data_set)
K = 5
B = np.ones((N,N))



for i in range(N):
    for j in range(N):
        dst = np.linalg.norm(data_set[i]-data_set[j])
        if(dst>1.25):
            B[i][j] = 0
        elif(i==j):
            B[i][j] = 0

for i in range(N): 
    for j in range(N): 
        if(B[i][j] == 1): 
            G.add_edge(i,j)

for i in range(N):
    for j in range(i+1,N):
        if(B[i][j]==1):
            xVals = [data_set[i][0],data_set[j][0]]
            yVals = [data_set[i][1],data_set[j][1]]
            plt.plot(xVals,yVals,'ko-',linewidth=1,markersize=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

W = nx.adjacency_matrix(G)
D = np.diag(np.sum(np.array(W.todense()), axis=1))
I = np.identity(B.shape[0])
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
L = I - np.dot(D_inv_sqrt, B).dot(D_inv_sqrt)
e,v = eigs(L, 5, which='SM')
centroids = v[[84,128,166,186,269],:]
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, data_set)
    if np.alltrue(centroids == old_centroids):
        break
    else:
        plt.figure(figsize = (12, 6))    
        plt.subplot(1, 2, 1)
        plot_current_state(centroids, memberships, data_set)

    old_memberships = memberships
    memberships = update_memberships(centroids, data_set)
    if np.alltrue(memberships == old_memberships):
        plt.show()
        break
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(centroids, memberships, data_set)
        plt.show()
    iteration = iteration + 1

