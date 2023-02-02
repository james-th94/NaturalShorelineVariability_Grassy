# -*- coding: utf-8 -*-
"""
Code to convert to MSL from LAT for bathy data.

Created on Thu Jan 26 13:12:46 2023

@author: s5245653
"""
# %% Change Directory
DIR = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/data/MetOcean"
import os
os.chdir(DIR)
# %% Import packages/modules
import pandas as pd # Working with tabular (2D+) data
import numpy as np # Working with array/vector data

# %% User inputs
MSL = 0.911 # metres LAT - from the tide data 1987 to 2021 (BOM)
fileIn = 'raw/Grassy_2021_GDA94_55_LAT.txt'
fileOut = 'processed/Grassy_2021_GDA94_55_MSL.xyz'

# %% Processing

# Load raw bathymetry data (in mLAT) to convert to mMSL.
bathyRaw = pd.read_csv(fileIn, delim_whitespace = True, header = None)

# Convert to MSL
bathyProcessed = bathyRaw.copy(deep = True)
bathyProcessed[2] = np.subtract(bathyProcessed[2], MSL)

# Save data - save depth file as an XYZ file with space separater
bathyProcessed.to_csv(fileOut, header = None, index = False, sep = ' ')