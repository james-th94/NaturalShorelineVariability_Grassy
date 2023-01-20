# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:01:39 2023

@author: s5245653
"""

# %% Change Directory
DIR = "C:/Users/s5245653/OneDrive - Griffith University/Projects/NaturalShorelineVariability_Grassy/data/MetOcean"
import os
os.chdir(DIR)
# %% Import packages/modules
import pandas as pd # Working with tabular (2D+) data
import numpy as np # Working with array/vector data

# %% Read data - elevation xyz data created in QGIS
elevation = pd.read_csv('processed/Bathy_30m_GA_2022/XYZ_Data_30m.xyz', delim_whitespace = True, header = None)

# %% Processing - create depth data (multiply elevation by -1)
depth = elevation.copy(deep = True)
depth[2] = np.multiply(depth[2], -1)

# %% Save data - save depth file as an XYZ file with space separater

depth.to_csv('processed/Bathy_30m_GA_2022/depth.xyz', header = None, index = False, sep = ' ')