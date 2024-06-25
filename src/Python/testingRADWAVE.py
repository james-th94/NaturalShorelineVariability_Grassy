# %%
import RADWave as rwave


# %%
downloads = 'C:/Users/s5245653/Downloads/'
file = 'testing_2.txt'
wa = rwave.waveAnalysis(altimeterURL = downloads + file,
                        bbox=[142.5,144.5,-42.0,-39.0],
                        stime=[1985,1,1], 
                        etime=[2022,12,31])
# %%
