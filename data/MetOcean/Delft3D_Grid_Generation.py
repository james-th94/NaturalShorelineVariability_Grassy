# -*- coding: utf-8 -*-
"""
Square grid generation in Delft3D format, allowing the user to specify the domain and size of each grid cell (in degrees).

User should edit the 'User Input' values in the script to get the result they want.


Created on Wed Jan 18 14:07:38 2023

@author: s5245653
"""

# %% Change Directory
DIR = "C:/Users/s5245653/Delft3D_Projects/Grassy_12012023/011"
import os
os.chdir(DIR)
# %% Import packages/modules
import numpy as np # Working with array/vector data
from datetime import datetime # Working with date/time variables

# %% Defining Functions

def CreateDelftGridFile(nGridPoints, gridStep, westLongitude, southLatitude, fileName, coordSys = 'Spherical'):
    # Create array of grid points - uses nGrid Points, gridStep, westLongitude and southLongitude
    lons = np.arange(0, nGridPoints) * gridStep + westLongitude
    lats = np.arange(0, nGridPoints) * gridStep + southLatitude
    # Convert grid point array to string for .grd file - uses nGridPoints, 
    lonstrings = [''] * nGridPoints
    latstrings = [''] * nGridPoints
    for j in range(nGridPoints):
        if j < 10:
            s = f' ETA=    {j+1}   '
            s2 = f' ETA=    {j+1}  '
        elif j < 99:
            s = f' ETA=    {j+1}  '
            s2 = f' ETA=    {j+1} '
        else:
            s = f' ETA=    {j+1} '
            s2 = f' ETA=    {j+1} '        
        for i in range(len(lons)):
            temp = lons[i]/100.0
            temp = '{:0.17f}'.format(temp)
            string = f'{temp}E+02   '
            temp2 = lats[j]/10.0
            temp2 = '{:0.17f}'.format(temp2)
            string2 = f'{temp2}E+01   '
            if ((i+1)%5) == 0:
                string += '\n             '
                string2 += '\n            '
            s += string
            s2 += string2
        lonstrings[j] = s
        latstrings[j] = s2
    # Create header line for .grd file - uses nGridPoints.
    now = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    header = f'*\n* Deltares, RGFGRID Version 5.05.00.59149 (Win64), Nov 13 2018, 21:19:05\n* File creation date: {now}\n*\nCoordinate System = {coordSys}\nMissing Value     =   -9.99999000000000024E+02\n      {nGridPoints}      {nGridPoints}\n 0 0 0\n'
    # Write to .grd file for Delft3D (Quickin or RFGRID)    
    file = open(fileName,'w')
    file.write(header)
    for a in lonstrings:
    	file.write(a+"\n")
    for b in latstrings:
        file.write(b+'\n')
    file.close()

# %% User Inputs

outFileCoarse = 'Grid_from_py_150m_KI.grd'
outFileFine = 'Grid_from_py_30m_KI.grd'
coordSystem = 'Spherical'

# Coarse Grid - 150 m
num = 100
step = 0.0015
lonsWest = float("{:.17f}".format(144.00025))
latsSouth = float("{:.17f}".format(-40.20005))

# Finer Grid - 30 m
# Finer grid needs to share a boundary with the coarser grid for SWAN modelling
numFine = 75 + 1 # Want a number here that is 1 more than a multiple of the ratio between the resolutions
stepFine = 0.0003 # Fine grid resolution in degrees (1 m = approx. 0.00001)
lonsWestFine = lonsWest + (30 * step) # Set the starting longitude at a grid point intersection of the coarse grid
latsSouthFine = latsSouth + (75 * step)

# Example code to run: 
CreateDelftGridFile(num, step, lonsWest, latsSouth, outFileCoarse)
CreateDelftGridFile(numFine, stepFine, lonsWestFine, latsSouthFine, outFileFine)
