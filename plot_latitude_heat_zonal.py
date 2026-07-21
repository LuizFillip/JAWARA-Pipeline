import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np 
 

def join_data():
    out = []
    for i in range(1, 5):
        infile = f'JAWARA/data/lat_7/T250{i}.txt'
         
        df = pd.read_csv(infile, index_col=0)
         
        ds = pd.pivot_table(
            df, 
            values = 't',
            index = 'altitude', 
            columns = 'time')
        out.append(ds)
        
    return pd.concat(out, axis = 1)


ds = join_data()
fig, ax = plt.subplots(
     figsize = (10, 6),
     dpi = 300
 )
 
 
 
ds.columns = pd.to_datetime(ds.columns)

ds = ds.loc[(ds.index > 20) & (ds.index < 120)]
contour = ax.contourf(
    ds.columns,
    ds.index,
    ds.values,
    levels=61,
    cmap="turbo",
    # norm=norm,
    # extend="both"
)

cbar = fig.colorbar(
    contour,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    r"$F_{\phi}$ (kg s$^{-2}$)"
)

ax.set_xlabel("Latitude (°)")
ax.set_ylabel("Altitude (km)")
 
ax.set_ylim(20, 120)
# ax.set_xlim(-90, 90)
