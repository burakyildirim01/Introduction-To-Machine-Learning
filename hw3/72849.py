import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

data_set = np.genfromtxt("hw03_data_set.csv", delimiter = ",", skip_header = 1)

x_train = data_set[:,0][:150]
y_train = data_set[:,1][:150]

x_test = data_set[150:272,0]
y_test = data_set[150:272,1]

N_test = y_test.shape[0]

bin_width = 0.37
origin = 1.5
minimum_value = np.min(data_set[:,0])-bin_width/3
maximum_value = np.max(data_set[:,0])+bin_width/3

data_interval = np.arange(minimum_value,maximum_value,0.01)

left_borders = np.arange(origin, maximum_value-bin_width, bin_width)
right_borders = np.arange(origin+bin_width, maximum_value, bin_width)
p_hat = np.asarray([np.sum(((left_borders[i] < x_train) & (x_train <= right_borders[i])) * y_train)/np.sum((left_borders[i] < x_train) & (x_train <= right_borders[i])) for i in range(len(left_borders))])

plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10,label='training')
plt.plot(x_test, y_test, "r.", markersize = 10,label='test')
plt.title('h = 0.37', fontweight='bold')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-") 
plt.legend()
plt.show()

regresso_total = 0
for i in range(len(left_borders)):
    for j in range(N_test):
        if(left_borders[i] < x_test[j] <= right_borders[i]):
            regresso_total += (y_test[j]-p_hat[i])**2

regresso_rmse = math.sqrt(regresso_total / N_test)
print("Regressogram => RMSE is %.4f when h is 0.37" % regresso_rmse)



p_hat_rms = np.asarray([np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) * y_train)/np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in data_interval])
plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10,label='training')
plt.plot(x_test, y_test, "r.", markersize = 10,label='test')
plt.plot(data_interval,p_hat_rms,'k-')
plt.title('h = 0.37', fontweight='bold')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


rms_test = np.asarray([np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) * y_train)/np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in x_test])
rms_rmse = np.sqrt(np.sum((y_test - rms_test)**2)/N_test)
print("Running Mean Smoother => RMSE is %.4f when h is 0.37" % rms_rmse)

p_hat_kernel = np.asarray([np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2) * y_train)/np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) for x in data_interval])
plt.figure(figsize = (10, 6))
plt.plot(x_train, y_train, "b.", markersize = 10,label='training')
plt.plot(x_test, y_test, "r.", markersize = 10,label='test')
plt.plot(data_interval,p_hat_kernel,'k-')
plt.title('h = 0.37', fontweight='bold')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend()
plt.show()


kernel_test = np.asarray([np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2) * y_train)/np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train)**2 / bin_width**2)) for x in x_test])
kernel_rmse = np.sqrt(np.sum((y_test - kernel_test)**2)/N_test)
print("Kernel Smoother => RMSE is %.4f when h is 0.37" % kernel_rmse)













