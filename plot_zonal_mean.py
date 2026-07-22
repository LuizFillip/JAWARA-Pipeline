import xarray as xr
import JAWARA as jw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import base as b 

b.sci_format()
def MERRA2_comp():
    from merra import load_merra, climatology_series, plot_climatology_bands
    
    ds = load_merra()
    
    
    
    clim_only, _, s_real = climatology_series(
        ds, "U_60N", 
        start_year = 2025, 
        start_month = 1, 
        start_day = 1, 
        end_month = 5, 
        end_year = 2025
        )
    
     
    ax_zon.plot(
        s_real.index,
        s_real.values,
        color="blue",
        lw=3.0,
        label= 'T, 60°-90° N'
    ) 
    
    clim_only, _, s_real = climatology_series(
        ds, "T_90N", 
        start_year = 2025, 
        start_month = 1, 
        start_day = 1, 
        end_month = 5, 
        end_year = 2025
        )
    
     
    ax_tem.plot(
        s_real.index,
        s_real.values,
        color="blue",
        lw=3.0,
        label= 'T, 60°-90° N'
    ) 


def concat_datasets(io, latitude = 60):
    datasets = []
    
    for month in range(1, 5):
        fn = rf"D:\database\JAWARA\{io}\{io}250{month}.nc"
    
        ds_month = (
            jw.load_data_netcdf(fn).sel(
                latitude = latitude, 
                method = "nearest"
                )
        .mean("longitude", skipna=True))
    
        datasets.append(ds_month)
     
    # Concatena janeiro–abril pelo tempo
    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )
    
    # Ordena e remove possíveis tempos duplicados
    ds = ds.sortby("time")
    
    _, idx = np.unique(
        ds.time.values,
        return_index=True,
    )
    
    
    return ds.isel(time=np.sort(idx))


da_zon = concat_datasets('U', latitude = 60)
da_tem = concat_datasets('T', latitude = 90)
 
 
#%%%%

def plot_time_height_diagram(ax, info, field):
    
  
    
    name = info['label']
    img = ax.contourf(
        field["time"],
        field["altitude"],
        field,
        levels= 31, 
        cmap= info['cmap'],  
        extend="both",
    )
    
    fig.colorbar(img, ax=ax, pad = 0.1)
     
    ax.set( 
        ylabel="Altitude (km)",
        ylim=(15, 120), 
        yticks = np.arange(10, 130, 20),
        
    )
    
    return None 



def avg_stratosphere(ax, field, name, altitude = 32):
    ax_right = ax.twinx()
    
    series_32km = field.sel(
        altitude = altitude,
        method = "nearest" 
        )
    
    ax_right.plot(
        series_32km.time,
        series_32km,
        color="blue",
        linewidth= 3,
        label=(
            f"{float(series_32km.altitude):.1f} km"
        )
    )
    
    b.change_axes_color(ax_right, "blue", side="right")
    
    ax_right.set_ylabel( "Average at 32 km (m/s)" )
    
    return ax_right 
    


def plot_field(ax, da, io):
    info = desc[io]
    
    field = da[io.lower()]
    
    field = (
        field.transpose("level", "time").swap_dims(
            {"level": "altitude"}
            ).sortby("altitude")
        )
     
    field = field.sel(altitude = slice(10, 120))
    
    ax_right =  plot_time_height_diagram(ax, info, field)
    
    ax_right = avg_stratosphere(ax, field, info['label'])
    
    return ax_right 
 
import datetime as dt 
from matplotlib.ticker import FuncFormatter

def format_doy(x, pos=None):
    data = mdates.num2date(x)
    return str(data.timetuple().tm_yday)
 
    
desc = {
    'V': {
        'label': "Meridional wind",
        'unit': '(m/s)', 
        'cmap': "RdBu_r"
        },
    'U': {'label': "Zonal wind", 
          'unit': '(m/s)', 
          'cmap': "RdBu_r"},
    'T': {'label': "Temperature", 
          'unit': '(K)', 
          'cmap': 'turbo'}
    }

import core as c 
df = c.low_omni()

fig, axes = plt.subplots(
    figsize = (16, 12),
    nrows = 3,
    sharex = True,
    dpi = 300,
)
 
plt.subplots_adjust(hspace = 0.2)
  
ax_tem = plot_field(axes[0], da_tem, io = 'T')
ax_zon = plot_field(axes[1], da_zon, io = 'U') 


ax_zon.axhline(0, linestyle = '--', color = 'blue')
ax_zon.set(ylim = [-70, 70])

names = [
    '(a) Polar temperature at 90°N', 
    '(b) Zonal mean of wind at 60°N',  
    ]

doy_ticks = np.arange(0, 121, 10)

tick_dates = [
    dt.datetime(2025, 1, 1) + dt.timedelta(days = int(doy - 1))
    for doy in doy_ticks
]

 
for i, ax in enumerate(axes): 
    onset = dt.datetime(2025, 2, 23)
    ax.axvline(onset, lw = 4,  color = 'green',  linestyle = '--')
    
    ax.text(0.01, 1.02, names[i], transform = ax.transAxes)
 
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(doy_ticks)

    ax.tick_params(
        axis="x",
        labelrotation=0
    )
     

axes[-1].set(xlabel = 'Day of Year (2025)')