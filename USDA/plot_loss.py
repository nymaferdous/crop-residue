import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
from pandas import read_excel
from tsmoothie.smoother import *
df = pd.read_csv("info.csv",index_col=2)
df1 = pd.read_csv("rand_permute.csv",index_col=2)
headers = ['Column1','Column2','Column3','Column4','Column5','Column6','Column7','Column8']
headers1 = ['Column1','Column2','Column3','Column4','Column5','Column6','Column7','Column8']

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
# df=pd.read_excel('/home/nyma/PycharmProjects/LOG.201106.xlsx')
print(df.head())
# x = df['Column3']
y = df['Column7']
y1 = df1['Column7']
smoother = ConvolutionSmoother(window_len=5, window_type='ones')
smoother1 = ConvolutionSmoother(window_len=5, window_type='ones')
smoother.smooth(y)
smoother1.smooth(y1)
low, up = smoother.get_intervals('sigma_interval', n_sigma=2)
low1, up1= smoother1.get_intervals('sigma_interval', n_sigma=2)
# smoother.smooth(y1)
# plot

# plt.plot(smoother.smooth_data[0], linewidth=1, color='blue', label="True Label")
plt.plot(y1, linewidth=1, color='g',label='Shuffle')
lgd=plt.legend(bbox_to_anchor=(1, 1))
# plt.plot(smooth(y,19), 'b-', lw=1)
# plt.plot(smoother.data[0], '.k')
# plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.2)
plt.fill_between(range(len(smoother1.data[0])), low1[0], up1[0], alpha=0.2)
# pylab.legend(loc='best')
# plt.plot(y)
plt.show()
# df.c.plot(color='r',lw=1.3)
# df.e.plot(color='Column5',lw=1.3)
# df = pd.read_excel("/home/nyma/PycharmProjects/LOG.201106.csv")
# df = np.genfromtxt(df, usecols=(2, 4), skip_header=2, dtype=None, encoding=None)
# x = df["C"]
# print(x)