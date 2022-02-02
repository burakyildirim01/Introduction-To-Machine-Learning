import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-100))

feature_set = np.genfromtxt("hw02_images.csv", delimiter = ",")
label_set = np.genfromtxt("hw02_labels.csv", delimiter = ",")


x_train = feature_set[:500]
x_test = feature_set[500:]

y_train = label_set[:500].astype(int)
y_test = label_set[500:].astype(int)

K = np.max(y_train)
N = x_train.shape[0]

Y_train = np.zeros((N, K)).astype(int)
Y_train[range(N), y_train - 1] = 1

def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

def gradient_W(X, Y_truth, Y_predicted):
    return np.matmul(np.transpose(X),((Y_truth - Y_predicted)*Y_predicted*(1 - Y_predicted)))

def gradient_w0(Y_truth, Y_predicted):
    return np.sum((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted))

eta = 0.0001
epsilon = 1e-3
max_iteration = 500

w = np.genfromtxt("initial_W.csv", delimiter = ",")
w0 = np.genfromtxt("initial_w0.csv", delimiter = ",")

iteration = 1
objective_values = []
while 1:
    Y_predicted = sigmoid(x_train, w, w0)

    objective_values = np.append(objective_values, 1/2*np.sum((Y_predicted - Y_train)**2))

    w_old = w
    w0_old = w0

    w = w + eta * gradient_W(x_train, Y_train, Y_predicted)
    w0 = w0 + eta * gradient_w0(Y_train, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((w - w_old)**2)) < epsilon:
        break
    if(iteration==max_iteration):
        break
    iteration += 1

plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

y_train_predicted = np.argmax(Y_predicted, axis = 1) + 1
cm = pd.crosstab(y_train_predicted, y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(cm)

y_test_predicted = np.argmax(sigmoid(x_test,w,w0),axis=1) + 1
cm_test = pd.crosstab(y_test_predicted, y_test, rownames = ['y_predicted'], colnames = ['y_test'])
print(cm_test)







