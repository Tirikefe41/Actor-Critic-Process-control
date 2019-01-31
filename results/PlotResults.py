import matplotlib.pyplot as plt
import numpy as np


r = np.load('/home/paperspace/Documents/chiller/Actor-Critic-Process-control/Data/Datasets.npz')
data = r['arr_0']

y0 = data[100:1000, 0]
y1 = data[100:1000, 1]
y3 = data[100:1000, 3]
y5 = data[100:1000, 5]

#Genrate figure and axis
fig1, axs = plt.subplots(3, 1)
axs[0].plot(y1, linewidth=0.5, color='b')
axs[0].set_ylabel('Power(Kw)')
axs[1].plot(y3, linewidth=0.5, color='b')
axs[1].set_ylabel('CHWRTemp(Celsius)')
axs[2].plot(y5, linewidth=0.5, color='b')
axs[2].set_ylabel('CWSTemp(Celsius)')
axs[2].set_xlabel('Time in minutes')


rdata =  np.load('/home/paperspace/Documents/chiller/Actor-Critic-Process-control/results/rewards_chiller_seCri.npz')
data1 = rdata['arr_0']
data2 = rdata['arr_1']
newdata = np.load('/home/paperspace/Documents/chiller/Actor-Critic-Process-control/results/total_rewards_chiller_seCri.npz')
data3 = newdata['arr_0']

fig2, axs2 = plt.subplots(3, 1)
axs2[0].plot(data1, linewidth=0.5)
axs2[0].set_ylabel('Average Reward')
axs2[1].plot(data2, linewidth=0.5)
axs2[1].set_ylabel('Average Qmax_Value')
axs2[2].plot(data3, linewidth=0.5)
axs2[2].set_xlabel('Number of Episodes')

fig3, axs3 = plt.subplots(1,1)
axs3[0].plot(data3, linewidth=0.5)
axs3[0].set_xlabel('Number of Episodes')
plt.show()