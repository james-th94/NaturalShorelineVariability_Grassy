# %% Code description
"""
Code to create SDS data frequency plot showing the change in 
frequency over the study period.
(e.g. more satellite imagery since 2015 due to Sentinel data)

Date created: 11/07/2023
"""

# %% Python setup

projDir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/"
import os

os.chdir(projDir)

# Import Packages and Functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
# from labellines import labelLines
import pickle

mpl.rcParams.update(mpl.rcParamsDefault)

# %% User inputs

inFile = projDir + "\\data\\RSP_raw.csv"
inFile2 = projDir + "\\data\\CoastSat\\Output\\Grassy_1980-01-01_2022-07-08_output.pkl"
outFig = projDir + "\\data\\Plots\\image_count_.png"
outFig2 = projDir + "\\data\\Plots\\image_count_stackedlines_.png"
outFig3 = projDir + "\\data\\Plots\\image_count_bar_.png"

L5_start = "16/09/1987"
L7_start = "07/07/1999"
L8_start = "17/04/2013"
S2_start = "18/11/2015"

# Formatting
colour = "grey"
plt.rcParams["text.color"] = colour
plt.rcParams["axes.labelcolor"] = colour
plt.rcParams["xtick.color"] = colour
plt.rcParams["ytick.color"] = colour
plt.rcParams["font.size"] = "14"
plt.gcf().autofmt_xdate()
resolution = 450

# %% Load satellite data
# Load from pickle file
pklFile = open(inFile2, "rb")
pkl = pickle.load(pklFile)
# Create dataframe for the data
dfSats = pd.DataFrame(
    index=pd.to_datetime(pkl["dates"]), data=pkl["satname"], columns=["Satellite"]
).sort_index()
# Count the yearly numbers for each satellite
dfSatsTotal = dfSats.resample("1YS").agg("count")
dfSatsTotal["L5"] = dfSats[dfSats["Satellite"] == "L5"].resample("1YS").agg("count")
dfSatsTotal["L7"] = dfSats[dfSats["Satellite"] == "L7"].resample("1YS").agg("count")
dfSatsTotal["L8"] = dfSats[dfSats["Satellite"] == "L8"].resample("1YS").agg("count")
dfSatsTotal["S2"] = dfSats[dfSats["Satellite"] == "S2"].resample("1YS").agg("count")
dfSatsTotal = dfSatsTotal.replace(np.NaN, 0)

# %% Plotting
# Plot as a stacked lineplot
fig, ax = plt.subplots(figsize=(10, 4))
labels = ["Landsat-5", "Landsat-7", "Landsat-8", "Sentinel-2"]
# colours = ["#636363", "#969696", "#bdbdbd", "#dfdfdf"]
ax.stackplot(
    dfSatsTotal.index,
    dfSatsTotal["L5"],
    dfSatsTotal["L7"],
    dfSatsTotal["L8"],
    dfSatsTotal["S2"],
#    colors=colours,
    labels=labels,
)
ax.plot(
    dfSatsTotal["Satellite"],
    color="k",
    lw=2,
    #        label = 'All satellites',
)
ax.legend(loc="upper left", fontsize="12")
ax.grid()
ax.set_xlim(dfSatsTotal.index[0], dfSatsTotal.index[-1])
ax.set_xlabel("Year")
plt.show()

# Save plot
# plt.savefig(outFig2, dpi=resolution, bbox_inches="tight")

# %% Bar plot

# Plot as a stacked bar graph
fig, ax = plt.subplots(figsize=(10, 4))
columns = ["L5", "L7", "L8", "S2"]
labels = ["Landsat-5", "Landsat-7", "Landsat-8", "Sentinel-2"]
colours = ["blue", "C0",  "#8eeebf",  "C2"]

# Landsat 5
ax.bar(dfSatsTotal.index.year, 
       height = dfSatsTotal.L5.values,
       color = colours[0],
       label = labels[0])
# Landsat 7
ax.bar(dfSatsTotal.index.year, 
       height = dfSatsTotal.L7.values,
       bottom = dfSatsTotal.L5.values,
       color = colours[1],
       label = labels[1])
# Landsat 8
ax.bar(dfSatsTotal.index.year, 
       height = dfSatsTotal.L8.values,
       bottom = (dfSatsTotal.L5.values + dfSatsTotal.L7.values),
       color = colours[2],
       label = labels[2])
# Sentinel 2
ax.bar(dfSatsTotal.index.year, 
       height = dfSatsTotal.S2.values,
       bottom = (dfSatsTotal.L5.values + dfSatsTotal.L7.values + dfSatsTotal.L8.values),
       color = colours[3],
       label = labels[3])

ax.legend(loc="upper left", fontsize="12")
ax.grid(linestyle = '--')
ax.set_xlim(1986,2021)
ax.set_xlabel("Year")
ax.set_ylabel("SDS Yearly Count")
# plt.show()

# Save plot
plt.savefig(outFig3, dpi=resolution, bbox_inches="tight")

# %% OLD

# %% Import data

dfRaw = pd.read_csv(inFile, parse_dates=True, index_col="dates")
df = dfRaw.resample("1Y").agg("count")

# %% Plot data as line plot

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
# Plot data
ax.plot(df.index, df["0"], linewidth=3)
# Plot dates of satellite images
ax.axvline(
    pd.to_datetime(S2_start, format="%d/%m/%Y"),
    ymin=0,
    ymax=0.2,
    color="C1",
    linewidth=3,
    label="S2",
)
ax.axvline(
    pd.to_datetime(L8_start, format="%d/%m/%Y"),
    ymin=0,
    ymax=0.2,
    color="C1",
    linewidth=3,
    label="L8",
)
ax.axvline(
    pd.to_datetime(L7_start, format="%d/%m/%Y"),
    ymin=0,
    ymax=0.2,
    color="C1",
    linewidth=3,
    label="L7",
)
ax.axvline(
    pd.to_datetime(L5_start, format="%d/%m/%Y"),
    ymin=0,
    ymax=0.2,
    color="C1",
    linewidth=3,
    label="L5",
)
# Label lines
labelLines(plt.gca().get_lines(), zorder=2.5)

# Axis and grid
ax.set_xlabel("Year")
ax.grid()

plt.savefig(outFig, dpi=resolution, bbox_inches="tight")


# %%
