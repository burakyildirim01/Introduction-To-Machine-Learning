import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.spatial as spa

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return memberships

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    plt.xlabel("x1")
    plt.ylabel("x2")

def em_e(data_set, centroids, covariances, epriors):
    priors = []
    for i in range(N):
        total = np.array([multivariate_normal(centroids[c], covariances[c]).pdf(data_set[i]) * epriors[c] for c in range(K)])
        n_sum = np.sum(total)
        p = np.array([(multivariate_normal(centroids[c], covariances[c]).pdf(data_set[i]) * epriors[c])/n_sum for c in range(K)])
        priors.append(p)
    return priors

def em_m(data_set, centroids, covariances, priors):
    epriors = []
    for c in range(K):
        pd = 0
        p = 0
        pdt = 0
        pr = 0
        for i in range(N):
            pd += priors[i][c] * data_set[i]
            p += priors[i][c]
            pdt += priors[i][c]*(np.dot(np.transpose(data_set[i].reshape(1,2) - centroids[c].reshape(1,2)),
                                       (data_set[i].reshape(1,2) - centroids[c].reshape(1,2))))
            pr += priors[i][c]
        covariances[c] = pdt/p
        centroids[c] = pd/p
        epriors.append(pr/N)
    return centroids, covariances, epriors

centroids = np.genfromtxt('hw05_initial_centroids.csv',delimiter=',')
data_set = np.genfromtxt('hw05_data_set.csv',delimiter=',',skip_header=1)
memberships = update_memberships(centroids, data_set)

N = data_set.shape[0]
K = 5
iteration = 100

priors = []
covariances = [np.cov(np.transpose(data_set[memberships == c])) for c in range(K)]
epriors = [data_set[memberships == c].shape[0] / N for c in range(K)]

for i in range(N):
    total = np.array([multivariate_normal(centroids[c], covariances[c]).pdf(data_set[i]) * (data_set[memberships == c].shape[0] / N) for c in range(K)])
    n_sum = np.sum(total)
    p = np.array([(multivariate_normal(centroids[c], covariances[c]).pdf(data_set[i]) * (data_set[memberships == c].shape[0] / N))/n_sum for c in range(K)])
    priors.append(p)


for i in range(iteration):
    centroids, covariances, epriors = em_m(data_set, centroids, covariances, priors)
    priors = em_e(data_set, centroids, covariances, epriors)
    print("Iteration: " + str(i+1))

result = np.vstack((centroids[0],centroids[2],centroids[1],centroids[4],centroids[3]))
print(result)

e_m = np.array([np.argmax(priors[c]) for c in range(N)])
plot_current_state(centroids, e_m, data_set)

x_interval, y_interval = np.mgrid[-6:6:0.05,-6:6:0.05]
position = np.empty(x_interval.shape + (2,))
position[:,:,0], position[:,:,1] = x_interval,y_interval

means = np.array(
    [[2.5, 2.5],
     [-2.5, 2.5],
     [-2.5, -2.5],
     [2.5, -2.5],
     [0, 0]])

deviations = np.array(
    [[[0.8, -0.6], [-0.6, 0.8]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[0.8, -0.6], [-0.6, 0.8]],
     [[0.8, 0.6], [0.6, 0.8]],
     [[1.6, 0], [0, 1.6]]])

covariances = np.array(covariances)

for c in range(K):
    gaussian_org = multivariate_normal(means[c], deviations[c])
    gaussian_em = multivariate_normal(centroids[c], covariances[c])
    plt.contour(x_interval, y_interval, gaussian_org.pdf(position), colors='k', levels=[0.05], linestyles='dashed')
    plt.contour(x_interval, y_interval, gaussian_em.pdf(position), colors='k', levels=[0.05])
    
plt.show()












