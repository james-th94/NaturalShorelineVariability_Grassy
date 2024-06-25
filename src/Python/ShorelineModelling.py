# %% Code description
"""
A shoreline model for Grassy Beach using the 2015 to 2021 satellite-derived shoreline (SDS) dataset.
Model will calculate shoreline change from wave, MSLP and SAMI inputs.
Goal is to predict shoreline position from 2021 to 2023 (WEC deployment period) to compare to SDS position. 
This will be used to assess impact of the nearshore wave energy converter (WEC) on the shoreline change.
RQ: Does the nearshore WEC influence shoreline change at Grassy Beach?

Input variables:
- CAWCR wave data (offshore) at (144.133, -40.066) - nearest to study site and captures easterly waves through Bass Strait:
    - significant wave height, wave period (mean and peak), wave direction (mean and peak), directional spread.
    - hourly dataset
    - calculate wave power as an additional variable
    - resample to monthly using median, mean and maximum.
- ERA5 (reanalysis) MSLP data at nearest node (144E, 40S):
    - MSLP (hPa)
    - daily dataset at 00 UTC timestep for simplicity (could download hourly data)
    - resample to monthly using minimum, as well as mean/median.
- SAM index data from British Antarctic Survey (BAS)
    - SAMI values
    - monthly dataset (no resampling required).

Response variables:
- Shoreline position 2015 to 2021 at 25 transects
    - resample to monthly (mean)
- Monthly shoreline change 2015 to 2021 at 25 transects

Method:
- Create large dataset with all data (approx. 20-30 columns per transect)
- Use df.corr() or a similar correlation method to quickly assess linear relationship between each input variable and the response variables
- Choose input variables based off previous step (i.e. does that input variable influence shoreline change)
- Subset large dataset based on input variables
- Split dataset at 2019/20 (2015 - 2019 and 2020 - 2021) into training (4 years) and test (2 years) datasets for model development (approx. 70/30 split)
- Build statistical/ machine-learning model to explain response variables from chosen input variables.
    - Various model choices available here...
    - Firstly, test on one transect (i.e. one timeseries)

Created by JT, 7 Nov 2023.
"""

# %% Python setup

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
