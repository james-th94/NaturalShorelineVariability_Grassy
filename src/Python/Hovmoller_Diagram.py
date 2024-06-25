# %% Code description
"""
Hovmoller plot (spatiotemporal visualisation) for shoreline change at Grassy 
from satellite-derived shoreline data (CoastSat results).

Also plot shoreline orientation bar graph with SAM timeseries,
and save csv data for statistical analysis 
(e.g., seasonal SAM to beach orientation).

Created: 23/04/2024
Last Modified: 23/04/2024 - version 7 plots
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
freq_monthly = 3
version = 8
maximum_shorelinechange = 40
startDate = pd.Timestamp("1987-Nov-30")
L5_start = pd.Timestamp("16/09/1987")
L7_start = pd.Timestamp("07/07/1999")
L8_start = pd.Timestamp("17/04/2013")
S2_start = pd.Timestamp("18/11/2015")

# SAM data
samFile = dataDir + "/MetOcean/raw/SAMIndex.csv"

# Plotting formatting
colour = "grey"
plt.rcParams["text.color"] = colour
plt.rcParams["axes.labelcolor"] = colour
plt.rcParams["xtick.color"] = colour
plt.rcParams["ytick.color"] = colour
plt.rcParams["font.size"] = "14"
plt.gcf().autofmt_xdate()
resolution = 450
outputFig = plotDir + f"Hovmoller_{freq_monthly}monthly_{version}.png"

# %% Load the data
# Shoreline (RSL/RSP/Shoreline anomaly) data
# Load the CSV file into a pandas DataFrame
dfRaw = pd.read_csv(dataDir + inputFile)
# Convert the "dates" column to datetime
dfRaw['dates'] = pd.to_datetime(dfRaw['dates'])
# Set "dates" as the index
dfRaw.set_index('dates', inplace=True)
df = dfRaw.resample(f'{freq_monthly}MS').mean()
df = df.truncate(before = startDate)

# SAM data
dfRawSAM = pd.read_csv(samFile)
dfRawSAM['DateTime'] = pd.to_datetime(dfRawSAM['DateTime'], format = "%d/%m/%Y")
dfRawSAM.set_index('DateTime', inplace = True)

dfSAM = dfRawSAM.truncate(before = startDate, after = "2020-12-31")
dfSAM = dfSAM.resample(f'{freq_monthly}MS').mean()
dfSAM['Month'] = dfSAM.index.month
dfSAM = Month2Season(dfSAM)

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
else:
    plt.yticks(original_yticks[0] + (12/freq_monthly)*np.arange(len(yrs)), # tick locations
            [f'{yr}-{df.index[0].month}' for yr in yrs], # tick labels 
            )
    plt.ylabel('Date (Year-Month)')

# Format plot and save figure
plt.tight_layout()
plt.savefig(outputFig, dpi = resolution, bbox_inches = "tight")

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


# Plot time series/ bar graph of beach orientation
for var in ['BOI', 'O']:
    fig, ax = plt.subplots(figsize = (10,6))
    # Add raw data
    # ax.scatter(df_Orientation.index, df_Orientation[var], 
    #         color = 'k', s = 10,)
    # ax.plot(df_Orientation.index, df_Orientation[var], color = 'k', lw = 2,)
    ax.bar(df_summer.index, df_summer[var],
           color = 'C3',
           width = freq_monthly*35,
           )
    ax.bar(df_winter.index, df_winter[var],
        color = 'C0',
        width = freq_monthly*35,
        )
    ax.bar(df_autumn.index, df_autumn[var],
        color = 'C4',
        width = freq_monthly*35,
        )
    ax.bar(df_spring.index, df_spring[var],
        color = 'C2',
        width = freq_monthly*35,
        )
    ax.axhline(0, color = 'k', lw =2)
    # Add seasonal mean values
    ax.axhline(df_summer[var].mean(), color = 'C3', 
               lw = 2, ls = '--', label = 'Summer (DJF)')
    ax.axhline(df_winter[var].mean(), color = 'C0', 
               lw = 2, ls = '--', label = 'Winter (JJA)')
    ax.axhline(df_autumn[var].mean(), color = 'C4', 
               lw = 2, ls = '--', label = 'Autumn (MAM)')
    ax.axhline(df_spring[var].mean(), color = 'C2', 
               lw = 2, ls = '--', label = 'Spring (SON)')
    # Format axes
    if var == 'BOI':
        ax.set_ylabel('Beach orientation')
    elif var == 'O':
        ax.set_ylabel('Beach orientation (m/m)')
    else:
        pass
    ax.set_xlabel('Year')
    ax.set_ylim(-1 * max(abs(df_Orientation[var].min()),df_Orientation[var].max())*1.05,
                max(abs(df_Orientation[var].min()),df_Orientation[var].max())*1.05)
    ax.grid()
    # Add legend
    ax.legend(loc = 'lower left', 
              ncol = 2)
    plt.tight_layout()
    plt.savefig(plotDir + f"RSL_{var}_{freq_monthly}monthly_{version}.png", dpi = resolution, bbox_inches = 'tight')

# Plot time series/ bar graph of beach orientation against SAM
for var in ['BOI', 'O']:
    fig, axs = plt.subplots(2,1, figsize = (10,6), sharex=True, height_ratios = [2,1])
    # Add raw data
    ax = axs[0]
    ax.bar(df_summer.index, df_summer[var],
           color = 'C3',
           width = freq_monthly*35,
           label = 'Summer (DJF)',
           )
    ax.bar(df_winter.index, df_winter[var],
        color = 'C0',
        width = freq_monthly*35,
        label = 'Winter (JJA)',
        )
    ax.bar(df_autumn.index, df_autumn[var],
        color = 'C4',
        width = freq_monthly*35,
        label = 'Autumn (MAM)',
        )
    ax.bar(df_spring.index, df_spring[var],
        color = 'C2',
        width = freq_monthly*35,
        label = 'Spring (SON)',
        )
    ax.axhline(0, color = 'k', lw =2)
    # Add seasonal mean values
    ax.axhline(df_summer[var].mean(), color = 'C3', 
               lw = 2, #label = 'Summer (DJF)', 
               ls = '--', )
    ax.axhline(df_winter[var].mean(), color = 'C0', 
               lw = 2, #label = 'Winter (JJA)', 
               ls = '--', )
    ax.axhline(df_autumn[var].mean(), color = 'C4', 
               lw = 2, #label = 'Autumn (MAM)', 
               ls = '--', )
    ax.axhline(df_spring[var].mean(), color = 'C2', 
               lw = 2, #label = 'Spring (SON)', 
               ls = '--', )
    # Format axes
    if var == 'BOI':
        ax.set_ylabel('Beach orientation')
    elif var == 'O':
        ax.set_ylabel('Beach orientation (m/m)')
    else:
        pass
    
    ax.set_ylim(-1 * max(abs(df_Orientation[var].min()),df_Orientation[var].max())*1.05,
                max(abs(df_Orientation[var].min()),df_Orientation[var].max())*1.05)
    ax.grid()
    ax.legend(loc = 'lower left', 
        ncol = 2)

    # Plot SAM values

    ax = axs[1]
    ax.bar(dfSAM[dfSAM['Season']=='Summer'].index, dfSAM[dfSAM['Season']=='Summer']['SAMI'],
           color = 'C3',
           width = freq_monthly*35,
           label = 'Summer'
           )
    ax.bar(dfSAM[dfSAM['Season']=='Winter'].index, dfSAM[dfSAM['Season']=='Winter']['SAMI'],
        color = 'C0',
        width = freq_monthly*35,
        label = 'Winter',
        )
    ax.bar(dfSAM[dfSAM['Season']=='Autumn'].index, dfSAM[dfSAM['Season']=='Autumn']['SAMI'],
        color = 'C4',
        width = freq_monthly*35,
        label = 'Autumn',
        )
    ax.bar(dfSAM[dfSAM['Season']=='Spring'].index, dfSAM[dfSAM['Season']=='Spring']['SAMI'],
        color = 'C2',
        width = freq_monthly*35,
        label = 'Spring',
        )
    ax.axhline(0, color = 'k', lw =2)
    ax.set_xlabel('Year')
    ax.grid()
    ax.set_ylabel('SAM')

    plt.tight_layout()
    plt.savefig(plotDir + f"RSLwSAM_{var}_{freq_monthly}monthly_{version}.png", dpi = resolution, bbox_inches = 'tight')

# Save data
# Data to save
a = dfSAM['SAMI'].values
b = df_Orientation['East'].values
c = df_Orientation['West'].values
d = df_Orientation['BOI'].values
e = df_Orientation['O'].values
f = dfSAM['Season'].values
# Combine into dataframe
tmp = pd.DataFrame({'SAMI': a, 'East_RSL':b, 'West_RSL':c,
                    'BOI':d, 'Orientation':e, 'Season':f},
                    index = dfSAM.index)
tmp = tmp.round(4) # Round all values to 4 decimal places
# Save to csv file
tmp.to_csv(dataDir + f"SAM_Orientation_Seasonal_{version}.csv")

# %% Statistics
df = tmp.copy(deep = True)
seasons = df.Season.unique()
df = df.dropna()
for col in ['BOI', 'Orientation', 'East_RSL', 'West_RSL']:
    for season in seasons:
        x = df[df['Season']==season]['SAMI']
        y = df[df['Season']==season][col]
        r = stats.pearsonr(x,y)
        # plt.scatter(x,y)
        # plt.show()
        print(f'For {season}, r_{col}={r}')


# No significant linear relationships,
# however a comparison of means shows that
# SAM > 0 winters have more eastern end erosion, and
# SAM > 0 summers have more western end erosion
# No significant influence on BOI though?
df[(df['SAMI']>0) & (df['Season'] == 'Winter')]



# %% THE END
# %% OLD
# %% Plot
sns.pairplot(data = df_Orientation[['West', 'East', 'Season']][df_Orientation['Season'].isin(['Summer','Winter'])], 
            hue = 'Season',
            palette = [sns.color_palette("tab10")[3], sns.color_palette("tab10")[0]],
            )

# %% Other

plt.scatter(df_Orientation[df_Orientation['Season']=='Summer']['East'], 
            df_Orientation[df_Orientation['Season']=='Summer']['West'])
plt.ylim(-22,22)
plt.xlim(-22,22)
plt.grid()
plt.show()

plt.scatter(df_Orientation[df_Orientation['Season']=='Winter']['East'], 
            df_Orientation[df_Orientation['Season']=='Winter']['West'])
plt.ylim(-22,22)
plt.xlim(-22,22)
plt.grid()
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(df_Orientation[df_Orientation['Season'].isin(['Winter','Summer'])]['East'], 
            df_Orientation[df_Orientation['Season'].isin(['Winter','Summer'])]['West'])
plt.ylim(-22,22)
plt.xlim(-22,22)
plt.plot(np.linspace(start = -22, stop = 22, num = 10),
         -1*np.linspace(start = -22, stop = 22, num = 10),
         color = 'k',
         ls = '--',
         )
plt.grid()
plt.show()


# %% THE END
