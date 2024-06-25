# %% Code description
"""
Code to create a large multiplot of climate and wave data for Grassy.
Similar to Figure 4 in Ibaceta et al. (2023) and to Bluecoast plots.

Date created: 06/08/2023

Environment: "datasci"
"""

# %% Python setup
# Import packages
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pickle as pkl # Saving and Loading dataframes for pandas
from datetime import datetime
import string
# Set directories
projDir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/"
os.chdir(projDir)
plotDir = projDir + "/data/Plots/"
# Plotting
mpl.rcParams.update(mpl.rcParamsDefault)
# Formatting
colour = "grey"
plt.rcParams["text.color"] = colour
plt.rcParams["axes.labelcolor"] = colour
plt.rcParams["xtick.color"] = colour
plt.rcParams["ytick.color"] = colour
plt.rcParams["font.size"] = "14"
plt.gcf().autofmt_xdate()
resolution = 450
# Constants
rho = 1026
g = 9.81

# Functions
def Month2Season(df):
    season_dict = {
        1: "Summer",
        2: "Summer",
        3: "Autumn",
        4: "Autumn",
        5: "Autumn",
        6: "Winter",
        7: "Winter",
        8: "Winter",
        9: "Spring",
        10: "Spring",
        11: "Spring",
        12: "Summer",
    }
    season_val_dict = {"Summer": 1.0, "Autumn": 2.0, "Winter": 3.0, "Spring": 4.0}
    df["Season"] = df["Month"].apply(lambda x: season_dict[x])
    df["Season_Val"] = df["Season"].apply(lambda x: season_val_dict[x])
    return df

# %% User inputs
# Input file locations
CAWCRFile = projDir + '/data/MetOcean/processed/CAWCR_Waves/ne_se_sw_Waves.pkl'
SAMFile = projDir + '/data/MetOcean/raw/SAMIndex.csv'

# Output file locations
outputFig = plotDir + '/Waves_SAM_3.png'

# Dates
startDate = pd.to_datetime("1987-03-01", format = "%Y/%m/%d")
endDate = pd.to_datetime("2020-12-31", format = "%Y/%m/%d")

# %% Import data
# CAWCR Wave data
dfWavesRaw = pd.read_pickle(CAWCRFile)
dfWaves = dfWavesRaw.sort_index().truncate(before = startDate, after = endDate)
dfWaves["power"] = np.divide(rho*np.power(g,2)*np.power(dfWaves["hs"],2)*(1/dfWaves["fp"]), 64000*np.pi)
dfWavesMonthly = dfWaves.resample('1MS').mean()
# SAM data
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')
dfSAM = pd.read_csv(SAMFile, parse_dates=["DateTime"], index_col="DateTime", 
                    date_parser = dateparse)

# %% Processing
df = dfSAM.join(dfWavesMonthly)
df = df.truncate(before = startDate, after = endDate)
hs = df.hsSE
tp = 1/df.fpSE
# Wave power
# (1/64 * pi) * rho * g^2 * Hs ^ 2 * Tp

# Get SAM positive and negative winter years
df["Month"] = df.index.month
df = Month2Season(df)
temp = df[(df["Season"] == 'Winter')].resample('3MS').mean()
negSAMYears = temp[temp["SAMI"] < 0].index.year
posSAMYears = temp[temp["SAMI"] > 0].index.year

# Subset data
dfPos = df[df.index.year.isin(posSAMYears)]
dfNeg = df[df.index.year.isin(negSAMYears)]
dfWinter = df[df["Season"]=="Winter"]
dfSummer = df[df["Season"]=="Summer"]

# %% Plotting

# %% SAMI only

# Create data variables
x = df.index
y = df.SAMI
yPos = df[df["SAMI"] > 0]['SAMI']
yNeg = df[df["SAMI"] < 0]['SAMI']
# Fit a trend line
x1 = np.arange(len(x))
trend = np.polyfit(x1,y,deg=1)

# Plot
fig, ax = plt.subplots(figsize = (10,6))
# Positive SAM only
ax.bar(yPos.index, yPos,
           color = 'C0',
           width = 50,
           alpha = 0.5,
           label = 'Positive SAMI')
# Negative SAM only
ax.bar(yNeg.index, yNeg,
           color = 'C1',
           alpha = 0.5,
           width = 50,
           label = 'Negative SAMI')
# All SAM
ax.plot(y.resample('3MS').mean(),
            color = '#343434',
            lw = 2,
            label = 'Seasonal mean')
# Long term trend
ax.plot(x, x1*trend[0] + trend[1],
        color = '#353935',
        lw = 1.5,
        ls = '--',
        label = 'Linear trend')

# Formatting
ax.legend(loc="lower right", fontsize="12")
ax.grid()
ax.set_xlim(x[0], x[-1])
ax.set_xlabel("Year")
# y-labels
ax.set_ylim(-1*np.max(np.abs(y)), np.max(np.abs(y)))
ax.set_ylabel("SAMI")

plt.plot()
# plt.savefig(outputFig, dpi = resolution, bbox_inches = "tight")

# %% Other

# %% Multiplot (Waves and SAM)

fig, axs = plt.subplots(2, sharex = True,
                        figsize = (10,12))
# Set variables
x = df.index
y1 = df.power
y3 = df.dpSE
y4 = df.SAMI
temp = dfWaves.resample('3MS').max()
y5 = temp[temp["hs"] > 4]
y4i = df[df["SAMI"] > 0]['SAMI']
y4ii = df[df["SAMI"] < 0]['SAMI']
# subplot 1: Wave height
axs[0].plot(x, y1,
           color = colour,
           lw = 1,
           label = 'Monthly mean')
axs[0].plot(y1.resample('3MS').mean(),
            color = 'k',
            lw = 3,
            label = 'Seasonal mean')
# axs[0].scatter(y5.index, df.loc[y5.index]["power"], color = 'C1')

# subplot 2: SAM index
axs[1].bar(y4i.index, y4i,
           color = 'C0',
           width = 50,
           label = 'Positive SAMI')
axs[1].bar(y4ii.index, y4ii,
           color = 'C1',
           width = 50,
           label = 'Negative SAMI')
axs[1].plot(y4.resample('3MS').mean(),
            color = 'k',
            lw = 3,
            label = 'Seasonal mean')
# Formatting
for ax in axs:
    ax.legend(loc="lower right", fontsize="12")
for ax in axs:
    ax.grid()
axs[-1].set_xlim(df.index[0], df.index[-1])
axs[-1].set_xlabel("Year")
# y-labels
axs[0].set_ylabel("Wave power (kW/m)")
axs[1].set_ylabel("SAMI")
# a,b,c
for n, ax in enumerate(axs):
    ax.text(0.02, 0.9, "(" + string.ascii_lowercase[n] + ")", transform=ax.transAxes,
            color = 'k', fontsize = 14,
            bbox=dict(facecolor='white', alpha=0.5, pad = 3))

plt.show()
# Save figure
# plt.savefig(outputFig, dpi = resolution, bbox_inches = 'tight')


# %% Plotting - OLD
fig, axs = plt.subplots(3, sharex = True,
                        figsize = (10,12))
# Set variables
x = df.index
y1 = power
y3 = df.dpSE
y4 = df.SAMI
y4i = df[df["SAMI"] > 0]['SAMI']
y4ii = df[df["SAMI"] < 0]['SAMI']
# subplot 1: Wave height
axs[0].plot(x, y1,
           color = 'C0',
           label = 'Monthly mean')
axs[0].plot(y1.resample('1Y').mean(),
            color = 'k',
            lw = 3,
            label = 'Yearly mean')
# subplot 2: Wave direction
axs[1].plot(x, y3,
            color = 'C0',
            label = 'Monthly mean')
axs[1].plot(y3.resample('1Y').mean(),
            color = 'k',
            lw = 3,
            label = 'Yearly mean')
# subplot 4: SAM index
axs[2].bar(y4i.index, y4i,
           color = 'C0',
           width = 50,
           label = 'Positive SAMI')
axs[2].bar(y4ii.index, y4ii,
           color = 'C1',
           width = 50,
           label = 'Negative SAMI')
axs[2].plot(y4.resample('1Y').mean(),
            color = 'k',
            lw = 3,
            label = 'Yearly mean')
# Formatting
for ax in axs:
    ax.legend(loc="lower right", fontsize="12")
for ax in axs:
    ax.grid()
axs[-1].set_xlim(df.index[0], df.index[-1])
axs[-1].set_xlabel("Year")
# y-labels
axs[0].set_ylabel("Wave power (kW/m)")
axs[1].set_ylabel("Peak wave direction (\u00b0)")
axs[2].set_ylabel("SAMI")
# a,b,c
for n, ax in enumerate(axs):
    ax.text(0.02, 0.9, "(" + string.ascii_lowercase[n] + ")", transform=ax.transAxes,
            color = 'k', fontsize = 14,
            bbox=dict(facecolor='white', alpha=0.5, pad = 3))

plt.show()
# Save figure
# plt.savefig(outputFig, dpi = resolution, bbox_inches = 'tight')

# %% THE END


# %%
