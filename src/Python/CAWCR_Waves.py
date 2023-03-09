# -*- coding: utf-8 -*-
"""
Code to Open and Compare the CAWCR wave data

Created on Wed Feb 22 2023

@author: s5245653
"""
# %% Change Directory
DIR = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/data/MetOcean"
import os
os.chdir(DIR)
# %% Import packages/modules
import pandas as pd # Working with tabular (2D+) data
import numpy as np # Working with array/vector data
import pickle as pkl # Saving and Loading dataframes for pandas

import matplotlib.pyplot as plt
from matplotlib import cm # https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
from windrose import WindroseAxes # https://windrose.readthedocs.io/en/latest/api.html
# %% User Inputs
files = ['raw/CAWCR_Waves/cawcr_aus_4m_KingsNE_197902_202212.csv', 
         'raw/CAWCR_Waves/cawcr_aus_4m_KingsSE_197902_202212.csv', 
         'raw/CAWCR_Waves/cawcr_aus_4m_KingsSW_197902_202212.csv']
idxCol = 'DateTime'
skpRows = 31
fileOut = 'processed/CAWCR_Waves/ne_se_sw_Waves.pkl'
figOut = 'processed/CAWCR_Waves/Comparison/_.png'
resolution = 300
# %% Location from files
def readLatLon(filename, rows_to_skip = 1, number_of_rows = 2):
    data = pd.read_csv(filename, skiprows = rows_to_skip, nrows = number_of_rows, header = None)
    return float(data[0][0][6:16]), float(data[0][1][6:16])

locations = []
for f in files:
    locations.append(readLatLon(f))
# %% Read data - Raw CAWCR data
# df = pd.read_csv(file, skiprows = skpRows, parse_dates = [idxCol], index_col = idxCol)
df = pd.DataFrame()
for f in files:
    dfTemp = pd.read_csv(f, skiprows = skpRows, parse_dates = [idxCol], index_col = idxCol)
    if df.empty:
        df = dfTemp
    else:
        df = df.join(dfTemp, rsuffix = f[34:36])
        
# Save to pickle file
df.to_pickle(fileOut)
# %% Waverose

# Plot Settings
# Waverose Settings
colorScale = None
binsRange = np.arange(0.01,7,1) # Controls the bins for the variable (e.g. Hs)
yRange = np.arange(10, 70, step=10) # Controls the percentage rings on the windrose
yTicks = []
for i in yRange:
    yTicks.append(str(i) + ' %')
sectors = 16
gap = 0.1 # value between 0 and 1 that defines the "gap" between rose sectors
edgeCol = 'grey'
compassValues = ['E', 'NE', 'N', 'NW',  'W', 'SW', 'S', 'SE']

# General Settings
plt.rcParams['font.size'] = '16'
fontsizeSmall = 12

# Legend Settings
roundingDec = 1
legTitle = 'Hs (m)'
labelLoc = 'lower right'

# Figure Output
#plotName = projectDir + plotDir + 'waverose.jpg'
#resolution = 600 # dpi

directionCol = 'dirSW'
heightCol = 'hsSW'

# PLOTTING WAVEROSE
ax = WindroseAxes.from_ax()
ax.bar(direction = df[directionCol], var = df[heightCol], bins = binsRange, nsector = sectors, 
       opening = 1.0 - gap, normed=True, edgecolor = edgeCol, cmap = colorScale)

# Plot a legend
ax.legend(title = legTitle, decimal_places = roundingDec, fontsize = fontsizeSmall, 
          loc = labelLoc)

# Fix the labelling
ax.set_xticklabels(compassValues, fontsize = fontsizeSmall)
ax.set_yticks(yRange)
ax.set_yticklabels(yTicks, fontsize = fontsizeSmall)

# Plot Figure
#plt.savefig(plotName, dpi = resolution, bbox_inches='tight')
plt.show()



# %% CDF wave heights

dfprobs = df.groupby('hs')['hs'].agg('count').pipe(pd.DataFrame).rename(columns = {'hs': 'frequency'})
dfprobs['cdf_NE'] = (dfprobs['frequency'] / sum(dfprobs['frequency'])).cumsum()
ax = dfprobs.plot(y = 'cdf_NE', grid = True)

dfprobs = df.groupby('hsSW')['hsSW'].agg('count').pipe(pd.DataFrame).rename(columns = {'hsSW': 'frequency'})
dfprobs['cdf_SW'] = (dfprobs['frequency'] / sum(dfprobs['frequency'])).cumsum()
dfprobs.plot(y = 'cdf_SW', grid = True, ax = ax)

dfprobs = df.groupby('hsSE')['hsSE'].agg('count').pipe(pd.DataFrame).rename(columns = {'hsSE': 'frequency'})
dfprobs['cdf_SE'] = (dfprobs['frequency'] / sum(dfprobs['frequency'])).cumsum()
dfprobs.index.names = ['hs']
dfprobs.plot(y = 'cdf_SE', grid = True, ax = ax)

plt.savefig(figOut, dpi = resolution, bbox_inches = 'tight')

# %% CDF wave directions

dfprobs = df.groupby('dir')['dir'].agg('count').pipe(pd.DataFrame).rename(columns = {'dir': 'frequency'})
dfprobs['cdf_NE'] = (dfprobs['frequency'] / sum(dfprobs['frequency'])).cumsum()
ax = dfprobs.plot(y = 'cdf_NE', grid = True)

dfprobs = df.groupby('dirSW')['dirSW'].agg('count').pipe(pd.DataFrame).rename(columns = {'dirSW': 'frequency'})
dfprobs['cdf_SW'] = (dfprobs['frequency'] / sum(dfprobs['frequency'])).cumsum()
dfprobs.plot(y = 'cdf_SW', grid = True, ax = ax)

dfprobs = df.groupby('dirSE')['dirSE'].agg('count').pipe(pd.DataFrame).rename(columns = {'dirSE': 'frequency'})
dfprobs['cdf_SE'] = (dfprobs['frequency'] / sum(dfprobs['frequency'])).cumsum()
dfprobs.index.names = ['dir']
dfprobs.plot(y = 'cdf_SE', grid = True, ax = ax)

plt.savefig(figOut, dpi = resolution, bbox_inches = 'tight')

