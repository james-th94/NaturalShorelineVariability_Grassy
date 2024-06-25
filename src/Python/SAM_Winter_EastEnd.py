# %% Code Description
"""

'datasci' Conda environment should be used.
"""
# %% Set Directories
projDir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/"
import os

os.chdir(projDir)

# %% Import Packages and Functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


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


# %% Data Files
negSAMData = (
    projDir + "data/SAM_neg.csv"
)  # Relative shoreline position data for negative SAM
posSAMData = (
    projDir + "data/SAM_pos.csv"
)  # Relative shoreline position data for positive SAM

# %% Shoreline Data - Read and Clean

dfNeg = pd.read_csv(negSAMData, index_col="dates", parse_dates=["dates"])
dfNeg["Month"] = dfNeg.index.month
dfNeg = Month2Season(dfNeg)

dfPos = pd.read_csv(posSAMData, index_col="dates", parse_dates=["dates"])
dfPos["Month"] = dfPos.index.month
dfPos = Month2Season(dfPos)

# %% Calculate Results
y1 = dfPos[dfPos["Season"] == "Winter"]["1200"]
y2 = dfNeg[dfNeg["Season"] == "Winter"]["1200"]
ttest_ind(y1, y2, nan_policy="omit")

# Result: Ttest_indResult(statistic=-1.2769104232880777, pvalue=0.20585345044749886)
# Positive SAM time periods have a smaller mean than negative SAM time periods

dfPlot = pd.DataFrame({"Positive SAM": y1, "Negative SAM": y2})
ax = sns.violinplot(dfPlot)
ax.set_ylabel("RSL (m)")
# %% Other figure

dataFile = projDir + "data/RSP_raw.csv"
df = pd.read_csv(dataFile, index_col="dates", parse_dates=["dates"])

colNames = df.columns
df["Month"] = df.index.month
df = Month2Season(df)

# %% Plotting
# Formatting
colour = "grey"
plt.rcParams["text.color"] = colour
plt.rcParams["axes.labelcolor"] = colour
plt.rcParams["xtick.color"] = colour
plt.rcParams["ytick.color"] = colour
plt.rcParams["font.size"] = "14"
plt.gcf().autofmt_xdate()

fig = plt.figure(figsize=(10, 6))
ax = sns.boxplot(df[colNames], color="grey", fliersize=0)
ax.plot(
    np.arange(0, len(colNames)),
    df[df["Season"] == "Summer"][colNames].median(),
    c="red",
    label="Summer",
    lw=3,
)
ax.plot(
    np.arange(0, len(colNames)),
    df[df["Season"] == "Winter"][colNames].median(),
    c="blue",
    label="Winter",
    lw=3,
)
# Summer Median RSL Positive and Negative SAM only
ax.plot(
    np.arange(0, len(colNames)),
    dfPos[dfPos["Season"] == "Summer"][colNames].median(),
    c="lightsalmon",
    lw=2,
    linestyle="dotted",
    label="Summer (SAM>0)",
)
ax.plot(
    np.arange(0, len(colNames)),
    dfNeg[dfNeg["Season"] == "Summer"][colNames].median(),
    c="crimson",
    lw=2,
    linestyle="dotted",
    label="Summer (SAM<0)",
)
# Winter Median Positive and Negative SAM only
ax.plot(
    np.arange(0, len(colNames)),
    dfPos[dfPos["Season"] == "Winter"][colNames].median(),
    c="lightblue",
    lw=2,
    linestyle="--",
    label="Winter (SAM>0)",
)
ax.plot(
    np.arange(0, len(colNames)),
    dfNeg[dfNeg["Season"] == "Winter"][colNames].median(),
    c="royalblue",
    lw=2,
    linestyle="--",
    label="Winter (SAM<0)",
)
ax.set_xticks(np.arange(0, len(colNames), 4))
ax.set_ylim(-50, 50)
ax.set_yticks([-40, -20, 0, 20, 40])
ax.set_xlabel("Distance Along Beach (m)")
ax.set_ylabel("RSL (m)")
ax.text(x=-3.8, y=-48, s="(Eroded)")
ax.text(x=-4, y=45, s="(Accreted)")
ax.text(x=-1.5, y=-60, s="(West)")
ax.text(x=23, y=-60, s="(East)")
ax.grid(True)
plt.legend(loc="best", 
           frameon = True,
           labelcolor="dimgrey",
           ncol = 3)
plt.savefig(projDir + "data\\Plots\\All\\boxplot_v2.png", dpi=600, bbox_inches="tight")


# %% Some statistics

seasonal_diff = df[df["Season"] == "Summer"][colNames].median()-df[df["Season"] == "Winter"][colNames].median()
quartiles = df.quantile([0.25,0.75])
iqr = quartiles.loc[0.75] - quartiles.loc[0.25]


# %%
