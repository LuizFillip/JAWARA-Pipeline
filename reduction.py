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

out = []
for num in tqdm(range(1, 5)):
    
    infile = f'JAWARA/data/T250{num}.txt'
    
    out.append(load_data(infile))

df = pd.concat(out)


df = df.set_index('time')

# df = df.resample('D').mean()




#%%%%%

ds = df.loc[
    (df['altitude'] == 100) & 
    (df.index.time == dt.time(0, 0))
    ].copy()

mat = pd.pivot_table(
    ds,
    values = 't',
    index = 'longitude',
    columns = ds.index
    )


mat.columns = mat.columns.dayofyear


def plot(mat):
    z = mat.values
    z = b.pass_band_2d(
        mat,
        bandpass = (2.2, 13)
    )


    



# 
import SSW25PW as p
import waves as wv
# ds1 = mat.loc[(mat.index == -33.7500)].T 

# time = ds1.index
# signal = ds1.values.flatten()

period_min, period_max = 2.1, 20
# filtered = b.bandpass_1d(
#     signal,
#     low_period= period_min,
#     high_period= period_max,
# )

filtered = b.pass_band_2d(
    mat,
    bandpass = (period_min, period_max)
)

fig, ax = plt.subplots()

# # result = p.plot_Wavelet(
# #     ax,
# #     time,
# #     filtered,
# #     period_min = period_min,
# #     period_max = period_max,
    
# # )start, end = 60, 90
# ds1 = ds.loc[:,
#     (ds.columns >= start) & 
#     (ds.columns <= end )
#     ].copy()


# wv.plot_zonalnumber_decomposition(
#         ax,  ds,
#         period_min = 2.5, 
#         period_max = 20,
#         y = 0.85,
#         x = 0.05,
#         fontsize = 30,
#         colorbar = False
#         )


vmax = np.nanpercentile( np.abs(filtered), 98 )

 
levels = np.linspace( -vmax, vmax, 30)
img = ax.contourf(
    mat.columns, 
    mat.index, 
    filtered, 
    levels = levels, 
    cmap = 'seismic'
    )
 
plt.colorbar(img)