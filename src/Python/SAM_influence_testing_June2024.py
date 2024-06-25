# %% Code description
"""
Code originally copied from Hovmoller plot code.

Created: 27/05/2024
Last Modified: 25/06/2024
Project: NaturalShorelineVariability
Python version: 3.10.9 (datasci environment)
"""

# %% Python setup
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os
import scipy.stats as stats

# Functions
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


# User inputs
projDir = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/"
plotDir = projDir + "/data/Plots/"
dataDir = projDir + "/data/"
os.chdir(projDir)

# Relative shoreline data
inputFile = "RSP_raw.csv"
freq_monthly = "NA" # seasonal?
version = 10
maximum_shorelinechange = 40
startDate = pd.Timestamp("1987-Feb-28")
L5_start = pd.Timestamp("16/09/1987")
L7_start = pd.Timestamp("07/07/1999")
L8_start = pd.Timestamp("17/04/2013")
S2_start = pd.Timestamp("18/11/2015")

# SAM data
samFile = dataDir + "/MetOcean/raw/SAMIndex.csv"

# Plotting formatting
colour = "grey"
plt.rcParams["text.color"] = 'k'
plt.rcParams["axes.labelcolor"] = colour
plt.rcParams["xtick.color"] = colour
plt.rcParams["ytick.color"] = colour
plt.rcParams["font.size"] = "14"
plt.gcf().autofmt_xdate()
resolution = 450
outputFig = plotDir + f"Testing_{freq_monthly}monthly_{version}.png"
outputFig2 = plotDir + f"Boxplot_ChangeInOrientation_{version}.png"

# %% Load the data
# Shoreline (RSL/RSP/Shoreline anomaly) data
# Load the CSV file into a pandas DataFrame
dfRaw = pd.read_csv(dataDir + inputFile)
# Convert the "dates" column to datetime
dfRaw['dates'] = pd.to_datetime(dfRaw['dates'])
# Set "dates" as the index
dfRaw.set_index('dates', inplace=True)
df = dfRaw.copy(deep = True)
# df = dfRaw.resample(f'{freq_monthly}MS').mean()
df = df.truncate(before = startDate)


# SAM data
dfRawSAM = pd.read_csv(samFile)
dfRawSAM['DateTime'] = pd.to_datetime(dfRawSAM['DateTime'], format = "%d/%m/%Y")
dfRawSAM.set_index('DateTime', inplace = True)

dfSAM = dfRawSAM.truncate(before = startDate, after = "2020-12-31")
# dfSAM = dfSAM.resample(f'{freq_monthly}MS').mean()
dfSAM['Month'] = dfSAM.index.month
dfSAM = Month2Season(dfSAM)

# %% "daily" Hovmoller - change the '1D' to '1MS' or '3MS' - change "freq" too.
empty_row = pd.DataFrame(columns=dfRaw.columns, index=[pd.to_datetime('1987-02-01 12:00:00+00:00')])
df = df.append(empty_row)
df = df.sort_index()
df_daily = df.resample('1D').mean()
# Create a new index with daily frequency from the start to the end date of your data
full_index = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='1D')

# Reindex the DataFrame to this new index to ensure all dates are present
df_reindexed = df_daily.reindex(full_index)

plt.figure(figsize=(10,12))
sns.heatmap(df_reindexed, 
            vmin=-0.6*maximum_shorelinechange, 
            vmax=0.6*maximum_shorelinechange, 
            cmap='coolwarm_r', 
            cbar_kws={'label': 'Shoreline anomaly (m)'},
            )
yrs = df.index.year.unique() # list of years for tick labels
plt.yticks(ticks=np.linspace(0, len(df_reindexed)-4, len(yrs)), 
           labels=yrs,           )
plt.show()



# %% OLD/OTHER:
# %% Hovmoller plot - choose the frequency in the User Inputs
# Create the hovmoller plot
plt.figure(figsize=(10,12))
sns.heatmap(df, 
            vmin=-0.6*maximum_shorelinechange, 
            vmax=0.6*maximum_shorelinechange, 
            cmap='coolwarm_r', 
            cbar_kws={'label': 'Shoreline anomaly (m)'},
            )

# Format axes
plt.xlabel('Alongshore distance from west (m)')
# Change y-ticks and labels
original_yticks = plt.gca().get_yticks()
yrs = df.index.year.unique() # list of years for tick labels
if freq_monthly == 3:
    plt.yticks(original_yticks[0] + (12/freq_monthly)*np.arange(len(yrs)), # tick locations
            [f'{yr}' + '/{:02d}'.format((int(str(yr)[-2:])+1) % 100) for yr in yrs], # tick labels
            )
    plt.ylabel('Date')
elif freq_monthly == 1:
    plt.yticks(original_yticks[0] + (12/freq_monthly)*np.arange(len(yrs)), # tick locations
            [f'{yr}-{df.index[0].month}' for yr in yrs], # tick labels 
            )
    plt.ylabel('Date (Year-Month)')
else:
    plt.ylabel('Date')

# Format plot and save figure
plt.tight_layout()
plt.show()
# plt.savefig(outputFig, dpi = resolution, bbox_inches = "tight")


# %% Beach orientation plots
df_Orientation = df.copy(deep=True)
df_Orientation[df_Orientation > maximum_shorelinechange] = maximum_shorelinechange
df_Orientation[df_Orientation < -1*maximum_shorelinechange] = -1*maximum_shorelinechange
df_Orientation['O'] = np.zeros(len(df_Orientation))
factor = -1

for i in range(len(df)):
    x = [int(j) for j in df.columns.values]
    y = df[df.index == df.index[i]].values[0]
    if np.isnan(sum(y)):
        slope = np.nan
    else:
        slope, intercept = np.polyfit(x, y, 1)
    df_Orientation['O'][i] = factor * slope

mean_orientation = df_Orientation['O'].mean()
std_orientation = df_Orientation['O'].std()
df_Orientation['BOI'] = np.divide(df_Orientation['O'] - mean_orientation, std_orientation)
df_Orientation['Month'] = df_Orientation.index.month
df_Orientation = Month2Season(df_Orientation)

df_summer = df_Orientation[df_Orientation.index.month.isin([12,1,2])]
df_winter = df_Orientation[df_Orientation.index.month.isin([6,7,8])]
df_spring = df_Orientation[df_Orientation.index.month.isin([9,10,11])]
df_autumn = df_Orientation[df_Orientation.index.month.isin([3,4,5])]

df_Orientation['West'] = df_Orientation[['0','50']].mean(axis=1)
df_Orientation['East'] = df_Orientation[['1150','1200']].mean(axis=1)
df_Orientation['West2'] = df_Orientation[['50','100','150']].mean(axis=1)
df_Orientation['East2'] = df_Orientation[['1100','1150','1200']].mean(axis=1)

df_Orientation['SAMI'] = dfSAM['SAMI'].values
df_Orientation['deltaBOI'] = df_Orientation['BOI'] - df_Orientation['BOI'].shift(2) # Shift 2 seasons to calculate change from summer to winter
df_Orientation['deltaO'] = df_Orientation['O'] - df_Orientation['O'].shift(2)

# Get results for SAMI > 0 and SAMI < 0:
# Summer
sum_pos = df_Orientation[(df_Orientation.index.month.isin([12]))&(df_Orientation['SAMI']>0)].describe()
sum_neg = df_Orientation[(df_Orientation.index.month.isin([12]))&(df_Orientation['SAMI']<0)].describe()
# Winter
wint_pos = df_Orientation[(df_Orientation.index.month.isin([6]))&(df_Orientation['SAMI']>0)].describe()
wint_neg = df_Orientation[(df_Orientation.index.month.isin([6]))&(df_Orientation['SAMI']<0)].describe()

# %% Plot data for post-S@ (end of 2015)
df_A = df_Orientation.truncate(before='2015-11-01')
east_transect = 'East'
west_transect = 'West'

# Winter
df_A_wint = df_A[df_A.index.month.isin([6])]
df_wint_pos = df_A_wint[df_A_wint['SAMI'] > 0]
df_wint_neg = df_A_wint[df_A_wint['SAMI'] < 0]
# Summer
df_A_sum = df_A[df_A.index.month.isin([12])]
df_sum_pos = df_A_sum[df_A_sum['SAMI'] > 0]
df_sum_neg = df_A_sum[df_A_sum['SAMI'] < 0]


fig, ax = plt.subplots(figsize = (10,6))
ax.plot(df_A[west_transect], color = 'k', label = 'West')
ax.plot(df_A[east_transect], color = 'gray', label = 'East')
# West - winter
ax.scatter(df_wint_pos.index, df_wint_pos[west_transect], color = 'C0', edgecolor = 'k', label = 'Winter (SAM > 0)')
ax.scatter(df_wint_neg.index, df_wint_neg[west_transect], color = 'C0', marker = 'x', edgecolor = 'k', label = 'Winter (SAM < 0)')
# West - summer
ax.scatter(df_sum_pos.index, df_sum_pos[west_transect], color = 'C3', edgecolor = 'k', label ='Summer (SAM > 0)')
ax.scatter(df_sum_neg.index, df_sum_neg[west_transect], color = 'C3', marker = 'x', edgecolor = 'k', label ='Summer (SAM < 0)')

# East - winter
ax.scatter(df_wint_pos.index, df_wint_pos[east_transect], color = 'C0', edgecolor = 'gray')
ax.scatter(df_wint_neg.index, df_wint_neg[east_transect], color = 'C0', marker = 'x', edgecolor = 'gray')
# East - summer
ax.scatter(df_sum_pos.index, df_sum_pos[east_transect], color = 'C3', edgecolor = 'gray')
ax.scatter(df_sum_neg.index, df_sum_neg[east_transect], color = 'C3', marker ='x', edgecolor = 'gray')


ax.set_xlabel('Year')
ax.set_ylabel('RSL (m)')
ax.set_ylim([-20,20])
ax.grid()
ax.legend(ncol = 3, loc = 'lower left')
plt.tight_layout()
plt.savefig(plotDir + f"RSL_{east_transect}_{west_transect}_{freq_monthly}monthly_{version}.png", dpi = resolution, bbox_inches = 'tight')

# %% Beach orientation (m/m) change between summer and winter
df = df_Orientation[['deltaO','Season','SAMI']]
temp = []
for i in df['SAMI'].values:
    if i >0:
        temp.append('Positive')
    elif i <0:
        temp.append('Negative')
    else:
        temp.append(np.NaN)
df['SAM'] = temp
df = df.drop(columns = ['SAMI'])
df = df.replace(['Summer', 'Winter'], ['Winter to\nSummer\n(Summer)', 'Summer\nto Winter\n(Winter)'])
# Plot results
fig, ax = plt.subplots(figsize = (10,6))
sns.boxplot(data = df[df['Season'].isin(['Winter to\nSummer\n(Summer)', 'Summer\nto Winter\n(Winter)'])], 
            y = 'Season', x = 'deltaO', hue = 'SAM', 
            orient = 'h', palette = ['C1','C0'], ax = ax, linewidth= 2, width = 0.5,
            )
ax.axvline(0,color = 'k')
ax.set_xlabel('Change in Beach Orientation ($\Delta$O)\n(negative is clockwise rotation, positive is anticlockwise)')
ax.set_xlim([-0.07,0.07])
ax.set_xticks(np.linspace(start = -0.06, stop = 0.06, num = 13))
ax.set_ylabel('')
ax.grid(axis = 'x', which = 'both')
ax.legend(loc = 'upper right', title = 'SAM Index:')
plt.tight_layout()
plt.savefig(outputFig2, dpi = resolution, bbox_inches = 'tight')

# %% THE END
