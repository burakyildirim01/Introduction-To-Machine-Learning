import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix

def safelog(x):
    return(np.log(x + 1e-100))

feature_set = pd.read_csv("hw01_images.csv", delimiter = ",", header = None).to_numpy()
label_set = pd.read_csv("hw01_labels.csv", header = None).to_numpy()

x_train = feature_set[:200]
x_test = feature_set[200:]

y_train = label_set[:200]
y_test = label_set[200:]

K = np.max(y_train)
N = x_train.shape[0]

male = []
female = []

for i in range(N):
    if(y_train[i]==1):
        female.append(x_train[i])
    else:
        male.append(x_train[i])

male = np.array(male)
female = np.array(female)


means = np.transpose(np.vstack((np.mean((female),axis=0),np.mean((male),axis=0))))

deviations = np.transpose(np.vstack((np.std((female),axis=0),np.std((male),axis=0))))

priors = [np.mean(y_train == (c+1)) for c in range(K)]

def get_score(x,mn,dev,prior):
    return np.sum((-0.5*safelog(2*math.pi*(dev**2))) -0.5*((x-mn)**2/(dev**2))) + safelog(prior)


y_train_predicted = []
y_test_predicted = []

for i in range(N):
    if(get_score(x_train[i],means[:,0],deviations[:,0],priors[0])>get_score(x_train[i],means[:,1],deviations[:,1],priors[1])):
        y_train_predicted.append(1)
    else:
        y_train_predicted.append(2)

for i in range(N):
    if(get_score(x_test[i],means[:,0],deviations[:,0],priors[0])>get_score(x_test[i],means[:,1],deviations[:,1],priors[1])):
        y_test_predicted.append(1)
    else:
        y_test_predicted.append(2)

print(confusion_matrix(y_train, y_train_predicted))

print(confusion_matrix(y_test, y_test_predicted))




