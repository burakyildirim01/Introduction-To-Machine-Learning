import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header = 1)

x_train = data_set[:,0][:150]
y_train = data_set[:,1][:150]

x_test = data_set[150:272,0]
y_test = data_set[150:272,1]

N = data_set.shape[0]
N_train = len(y_train)
N_test = len(y_test)

P = 25

def regression_dt(P, N_train, x_train, y_train):
    
  node_indices = {}
  is_terminal = {}
  need_split = {}
  
  node_splits = {}
  
  node_indices[1] = np.array(range(N_train))
  is_terminal[1] = False
  need_split[1] = True
  
  while True:
      
    split_nodes = [key for key, value in need_split.items() if value == True]
   
    if len(split_nodes) == 0:
      break
  
    for split_node in split_nodes:
      data_indices = node_indices[split_node]
      need_split[split_node] = False
      
      if len(data_indices) <= P:
        node_splits[split_node] = np.mean(y_train[data_indices])
        is_terminal[split_node] = True
      else:
        is_terminal[split_node] = False
        unique_values = np.sort(np.unique(x_train[data_indices]))
        split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
        split_scores = np.repeat(0.0, len(split_positions))
        
        for s in range(len(split_positions)):
          error = 0
          left_indices = data_indices[x_train[data_indices] < split_positions[s]]
          right_indices = data_indices[x_train[data_indices] >= split_positions[s]]
          left_mean = np.mean(y_train[left_indices])
          right_mean = np.mean(y_train[right_indices])
          error += np.sum((y_train[left_indices]-left_mean)**2) + np.sum((y_train[right_indices] - right_mean) ** 2)
          split_scores[s] = error / (len(left_indices) + len(right_indices))
          
        best_splits = split_positions[np.argmin(split_scores)]
        node_splits[split_node] = best_splits
        
        left_indices = data_indices[x_train[data_indices] < best_splits]
        node_indices[2 * split_node] = left_indices
        is_terminal[2 * split_node] = False
        need_split[2 * split_node] = True
        
        right_indices = data_indices[x_train[data_indices] >= best_splits]
        node_indices[2 * split_node + 1] = right_indices
        is_terminal[2 * split_node + 1] = False
        need_split[2 * split_node + 1] = True
        
  return is_terminal, node_splits

def predict(data, node_splits, is_terminal):
    N = data.shape[0]
    y_predicted = np.zeros(N)
    for i in range(N):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted[i] = node_splits[index]
                break
            else:
                if data[i] <= node_splits[index]:
                    index *= 2
                else:
                    index *= 2
                    index += 1
    return y_predicted

def calculate_rmse(y_truth, y_pred):
    return np.sqrt(np.mean((y_truth-y_pred)**2))


is_terminal = regression_dt(P, N_train, x_train, y_train)[0]
node_splits = regression_dt(P, N_train, x_train, y_train)[1]

data_interval = np.linspace(1.5, 5.25, 500)
y_predicted = predict(data_interval, node_splits, is_terminal)

plt.figure(figsize = (15, 6))
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.title("P = 25",fontweight='bold')
plt.plot(x_train,y_train,"b.", markersize = 10,label="training")
plt.plot(x_test,y_test,"r.", markersize = 10,label="test")
plt.plot(data_interval, y_predicted, "k-")
plt.legend()
plt.show()

test_values = predict(x_test, node_splits, is_terminal)
rmse = calculate_rmse(y_test, test_values)
print("RMSE is %.4f when P is 25" % rmse)

rmse_values = [0] * 10
p_values = np.arange(5, 55, 5)
for i in range(len(p_values)):
    is_terminal_temp = regression_dt(p_values[i], N_train, x_train, y_train)[0]
    node_splits_temp = regression_dt(p_values[i], N_train, x_train, y_train)[1]
    rmse_values[i] = calculate_rmse(y_test, predict(x_test, node_splits_temp, is_terminal_temp))

plt.figure(figsize = (15, 6))
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.plot(p_values, rmse_values, "ko-", markersize = 10)
plt.show()







