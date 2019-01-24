import numpy as np
import matplotlib.pyplot as plt

file = 'critic_notargetsq.npz'

df0 = np.load(file)['arr_0']
df1 = np.load(file)
df2 = np.load(file)
##df0.shape
print('df0',df0)
##print('df1',df1)
##print('df2',df2)

##plt.ion()
##plt.figure()
##plt.imshow(df[0])
