# %% Code Description
"""
Code to transform shoreline data locations along transect lines 
into x,y points in EPSG 3857 Coordinate Reference System for GIS.

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
import json
import pandas as pd


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
transectFile = projDir + "data/CoastSat/Input/Transects_Grassy.geojson"
with open(transectFile) as f:
    data = json.load(f)

slFile = (
    projDir
    + "data/CoastSat/Output/Grassy_1980-01-01_2022-07-08_shorelines_tidecorrected.csv"
)

# %% Shoreline Data - Read and Clean

df = pd.read_csv(slFile, index_col="dates", parse_dates=["dates"])
df["Month"] = df.index.month
df = Month2Season(df)
df = df.drop(columns=["Month", "Season_Val"])

dfSummer = df[df["Season"] == "Summer"]
dfWinter = df[df["Season"] == "Winter"]

df = df.drop(columns=["Season"])
dfSummer = dfSummer.drop(columns=["Season"])
dfWinter = dfWinter.drop(columns=["Season"])

# Calculate Statistical Values
winter = [dfWinter[col].quantile(0.5) for col in dfWinter.columns]
summer = [dfSummer[col].quantile(0.5) for col in dfSummer.columns]
mins = [df[col].quantile(0) for col in df.columns]
q1 = [df[col].quantile(0.25) for col in df.columns]
medians = [df[col].quantile(0.5) for col in df.columns]
q3 = [df[col].quantile(0.75) for col in df.columns]
maxs = [df[col].quantile(1) for col in df.columns]


# Create coordinate lists from json file
coords0 = [feature["geometry"]["coordinates"][0] for feature in data["features"]]
coords1 = [feature["geometry"]["coordinates"][1] for feature in data["features"]]
names = [feature["properties"]["name"] for feature in data["features"]]
transects = range(len(coords0))

# Get data points from the geojson file for transects
x0 = [coords0[trans][0] for trans in transects]
x1 = [coords1[trans][0] for trans in transects]
y0 = [coords0[trans][1] for trans in transects]
y1 = [coords1[trans][1] for trans in transects]

# Calculate the angle of the transect, including the quadrant
angles = [np.arctan2(y1[i] - y0[i], x1[i] - x0[i]) for i in transects]

# Get distance (metres) along the line from the CoastSat output
# [winter, summer, Q1, median, Q3, summer, winter]
# blue, orange, green, red, purple.
dists = [
    [winter[i], summer[i], mins[i], q1[i], medians[i], q3[i], maxs[i]]
    for i in transects
]
vals = range(len(dists[0]))

# Create new data points (in EPSG 3857 coordinates (metres from 0N,0E))
x_new = [[x0[i] + dists[i][j] * np.cos(angles[i]) for j in vals] for i in transects]
y_new = [[y0[i] + dists[i][j] * np.sin(angles[i]) for j in vals] for i in transects]

# Plot to check the point is along the line
# for i in transects:
#     fig, ax = plt.subplots()
#     ax.scatter(x0[i], y0[i], c="k", marker=".")
#     ax.scatter(x1[i], y1[i], c="k", marker="*")
#     for j in vals:
#         ax.scatter(x_new[i][j], y_new[i][j], c=f"C{j}")
#     plt.text(x=x1[i], y=y1[i], s=names[i])
#     plt.plot()

# %% Create Output Data
winterSL = pd.DataFrame(data=[np.array(x_new)[:, 0], np.array(y_new)[:, 0]]).transpose()
winterSL.columns = ["x", "y"]
winterSL.index = np.arange(1200, -50, -50)
# winterSL.to_csv(projDir + "data/CoastSat/Output/Shorelines/winterSL.csv")

summerSL = pd.DataFrame(data=[np.array(x_new)[:, 1], np.array(y_new)[:, 1]]).transpose()
summerSL.columns = ["x", "y"]
summerSL.index = np.arange(1200, -50, -50)
# summerSL.to_csv(projDir + "data/CoastSat/Output/Shorelines/summerSL.csv")

minSL = pd.DataFrame(data=[np.array(x_new)[:, 2], np.array(y_new)[:, 2]]).transpose()
minSL.columns = ["x", "y"]
minSL.index = np.arange(1200, -50, -50)
# minSL.to_csv(projDir + "data/CoastSat/Output/Shorelines/minSL.csv")

q1SL = pd.DataFrame(data=[np.array(x_new)[:, 3], np.array(y_new)[:, 3]]).transpose()
q1SL.columns = ["x", "y"]
q1SL.index = np.arange(1200, -50, -50)
# q1SL.to_csv(projDir + "data/CoastSat/Output/Shorelines/q1SL.csv")

medianSL = pd.DataFrame(data=[np.array(x_new)[:, 4], np.array(y_new)[:, 4]]).transpose()
medianSL.columns = ["x", "y"]
medianSL.index = np.arange(1200, -50, -50)
# medianSL.to_csv(projDir + "data/CoastSat/Output/Shorelines/medianSL.csv")

q3SL = pd.DataFrame(data=[np.array(x_new)[:, 5], np.array(y_new)[:, 5]]).transpose()
q3SL.columns = ["x", "y"]
q3SL.index = np.arange(1200, -50, -50)
# q3SL.to_csv(projDir + "data/CoastSat/Output/Shorelines/q3SL.csv")

maxSL = pd.DataFrame(data=[np.array(x_new)[:, 6], np.array(y_new)[:, 6]]).transpose()
maxSL.columns = ["x", "y"]
maxSL.index = np.arange(1200, -50, -50)
# maxSL.to_csv(projDir + "data/CoastSat/Output/Shorelines/maxSL.csv")

# %% summer to winter distance

diffSeasons = summerSL - winterSL
diffSeasons["Distance"] = np.sqrt(
    np.power(diffSeasons.x, 2) + np.power(diffSeasons.y, 2)
)
# %%
