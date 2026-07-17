import xarray as xr 
import glob 
import numpy as np 
import base as b 

import pandas as pd 



df = pd.read_csv('JAWARA/data/T2501.txt', index_col = 0)


#%%%%%

df['altitude'] = df['altitude'].astype(int)

ds = df.loc[df['altitude'] == 100]

ds.loc[:, "time"] = pd.to_datetime(ds['time'])

ds.loc[:, 'longitude'] = ((ds['longitude'] + 180) % 360) - 180
import matplotlib.pyplot as plt 


ds1 = pd.pivot_table(
    ds,
    values = 't',
    index = 'longitude',
    columns = 'time'
    )


plt.contourf(
    ds1.columns, 
    ds1.index, 
    ds1.values, 
    levels = 30, 
    cmap = 'jet'
    )
 