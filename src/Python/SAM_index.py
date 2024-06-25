# %% Python setup
import pandas as pd
import os
import matplotlib.pyplot as plt

projDir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/"
plotDir = projDir + "/data/Plots/"
os.chdir(projDir)

# Plotting formatting

colour = "grey"
plt.rcParams["text.color"] = colour
plt.rcParams["axes.labelcolor"] = colour
plt.rcParams["xtick.color"] = colour
plt.rcParams["ytick.color"] = colour
plt.rcParams["font.size"] = "14"
plt.gcf().autofmt_xdate()
resolution = 450

# %% Import data
samiLocation = projDir + "/data/MetOcean/raw/SAMIndex.csv"
dfRaw = pd.read_csv(samiLocation, parse_dates=True, index_col="DateTime")
df = dfRaw.truncate(after = "2020-12-31")
dfYearly = df.resample('1YS').mean()

outputFig = plotDir + "SAMI.png"

# %% Plot data
fig, ax  = plt.subplots(figsize = (15,8))
# SAMI - scatter plot
ax.scatter(df.index, df.SAMI,
           s = 10, color = 'grey',
           label = 'Monthly average')
# Yearly average - line plot
ax.plot(dfYearly.index, dfYearly.SAMI,
        lw = 3, color = 'C0',
        label = 'Yearly average')
# Formatting
ax.axhline(0, color = 'k', lw = 1)
ax.grid()
ax.legend()

ax.set_ylim([-5, 5])
ax.set_ylabel('SAMI')
ax.set_xlabel('Year')

plt.savefig(outputFig, dpi = resolution)

# %%
