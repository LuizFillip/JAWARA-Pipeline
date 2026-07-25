import xarray as xr
import JAWARA as jw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import base as b 
import datetime as dt  
from indices import plot_kp_by_disturbed_level 

import core as c 
b.sci_format()
def MERRA2_comp():
    from merra import load_merra, climatology_series 
    
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





da_zon = jw.concat_datasets('U', latitude = 60)
da_tem = jw.concat_datasets('T', latitude = 90)
 
 
#%%%%

def plot_time_height_diagram(ax, info, field):
    
 
    name, unit = info['label'], info['unit']
    img = ax.contourf(
        field["time"],
        field["altitude"],
        field,
        levels= 31, 
        cmap= info['cmap'],  
        extend="both",
    )
    
    cb = plt.colorbar(img, ax=ax, pad = 0.01)
    cb.set_label(f'{name} {unit}')
    
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


def plot_ssw_parameters_and_kp():

    
    fig, axes = plt.subplots(
        figsize = (16, 12),
        nrows = 3,
        sharex = True,
        dpi = 300,
        constrained_layout=True
    )
     
    plt.subplots_adjust(hspace = 0.2)
      
    ax_tem = plot_field(axes[0], da_tem, io = 'T')
    ax_zon = plot_field(axes[1], da_zon, io = 'U') 
    
    
    ax_zon.axhline(0, linestyle = '--', color = 'blue')
    
    ax_zon.set(
        ylim = [-60, 80], 
        yticks = np.arange(-60, 70, 10)
        )
    
    
    
    doy_ticks = np.arange(0, 121, 10)
    
    tick_dates = [
        (dt.datetime(2025, 1, 1) + 
         dt.timedelta(days = int(doy - 1)))
        for doy in doy_ticks
    ]
    
    xmin = dt.datetime(2025, 1, 1)
    xmax = dt.datetime(2025, 4, 30)
    plot_kp_by_disturbed_level( axes[-1], c.low_omni(), xmin, xmax  )
    
    names = [
        '(a) JAWARA temperature at 90°N', 
        '(b) JAWARA zonal wind at 60°N',  
        '(c) Kp index' 
        ]
    
    
    
    for ax in [axes[0], axes[1]]:
        ax.set_xlim(xmin, xmax)
        
        
    for i, ax in enumerate(axes): 
        onset = dt.datetime(2025, 2, 23)
        ax.axvline(
            onset, lw = 4, 
            color = 'green', 
            linestyle = '--', 
            label = 'SSW onset')
        
        ax.text(0.01, 1.02, names[i], transform = ax.transAxes)
     
        ax.set( 
            xticks = tick_dates, 
            xticklabels = doy_ticks, 
            
            )
        
        ax.set_xlim(xmin,  xmax )
        
        
        ax.tick_params(
            axis="x",
            labelrotation=0
        )
        
      
    
    axes[-1].set(xlabel = 'Day of Year (2025)' ) 
    return fig
    
