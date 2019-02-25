import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


r = np.load('C:/Users/ANIBOH OLAIDE/Documents/UPWORK/4 DRL Encode/Juan/Chillercodes_backup/Data/Datasets.npz')
data = r['arr_0']



pdata = data[0:1440,1]
cldata = data[0:1440,0]
df = pd.DataFrame({'power':pdata, 'load':cldata})
mn = min(df['power'])
df = df[df.power > mn]
print(df.index)

plt.plot(df['load'], df['power'],'.', alpha = 0.7, label = 'Test Data')

x_min = []
y_min = []
width = 4

for i in range(0, len(df), width):

	_df = df[(df.index > i) & (df.index < i + width)] # vertical slice
	# print(_df)
	ddf = _df.loc[_df['power'].idxmin()]
	# print(ddf)
	ddf.dropna()
	x_min.append(ddf.load)                         
	y_min.append(ddf.power)

print("X values are: {}".format(x_min))
print("Y values are: {}".format(y_min))
df2 = pd.DataFrame({'X': x_min, 'Y': y_min})
df2.sort_values(by=['X'])

from scipy.optimize import leastsq
# guesses =[1000, 100, -100]
# fit_pars, flag = leastsq(func = flipped_resid, x0 = guesses,
#                          args = (pdata, cldata))
# # plot the fit:
# y_fit = model(x_data, *fit_pars)
# y_guess = model(x_data, *guesses)
# plt.plot(x_data, y_fit, 'r-', zorder = 0.9, label = 'Envelope Plot')
# plt.legend(loc = 'lower left')
plt.show()