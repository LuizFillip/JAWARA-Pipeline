import xarray as xr 
from tqdm import tqdm 
import numpy as np 
import base as b 
import datetime as dt 
import matplotlib.pyplot as plt 
import pandas as pd 



def load_data(infile):
    ds = pd.read_csv(infile, index_col = 0)
    
    ds['altitude'] = ds['altitude'].astype(int)
 
    ds.loc[:, "time"] = pd.to_datetime(ds['time'])
    
    ds.loc[:, 'longitude'] = ((ds['longitude'] + 180) % 360) - 180
    
    return ds 

def join_data():

    out = []
    io = 'Joining'
    for num in tqdm(range(1, 5), io):
        
        infile = f'JAWARA/data/T250{num}.txt'
        
        out.append(load_data(infile))
    
    df = pd.concat(out)
    df["time"] = pd.to_datetime(df["time"])

    df = df.set_index("time") 
    return df

df = join_data()



#%%%%
 

ds = df.loc[ 
    (df.index.time == dt.time(0, 0))
    ].copy()


def plot_temp_by_lon_doy(ds):
    fig, ax = plt.subplots(
        ncols = 3,
        nrows = 3, 
        figsize = (14, 14)
        )
    
    alts = np.arange(20, 110, 10)
    
    axes = ax.flat
    for i, alt in enumerate(alts):
        ds2 = ds.loc[ds['altitude'] == alt].copy()
        ds2.index = ds2.index.dayofyear
        
        mat = pd.pivot_table(
            ds2,
            values = 't',
            index = 'longitude',
            columns = ds2.index
            )
    
        levels = np.linspace(180, 350, 30)
        axes[i].set(title = f'{alt} km')
        img = axes[i].contourf(
             mat.columns, 
             mat.index, 
             mat.values,
             levels = levels, 
             cmap = 'turbo'
             )
        
        fig.colorbar(img)
        
        
plot_temp_by_lon_doy(ds)

# ds['t'].plot(kind = 'hist')