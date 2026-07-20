import xarray as xr 

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

def plot_time_altitude(da_plot, da_lat):
    # limit = float(
    #     np.nanpercentile(
    #         np.abs(da_plot.values),
    #         98
    #     )
    # )
    
    # norm = TwoSlopeNorm(
    #     vmin=-limit,
    #     vcenter=0,
    #     vmax=limit
    # )
    
    fig, ax = plt.subplots(
        figsize=(13, 6)
    )
    
    contour = ax.contourf(
        da_plot["time"],
        da_plot["z_m"] / 1000.0,
        da_plot['acceleration_day'],
        levels=31,
        cmap="RdBu_r",
        # norm=norm,pl
        extend="both"
    )
    
    colorbar = fig.colorbar(
        contour,
        ax=ax,
        pad=0.02
    )
    
    colorbar.set_label(
        r"EP-flux forcing "
        r"(m s$^{-1}$ day$^{-1}$)"
    )
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Altitude (km)")
    
    ax.set_title(
        rf"EP-flux forcing at "
        rf"{float(da_lat.latitude):.2f}°"
    )

    ax.set_ylim(
        20,
        140
    )
    
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

import datetime as dt 
    
ep = xr.open_dataset('JAWARA/data/zonal_mean/ep_flux_2503.nc')

da_lat = (
    ep
    .sel(
        latitude=-7,
        method="nearest"
    )
    # .sel(
    #     time=slice(
    #         "2025-02-13",
    #         None
    #     )
    # )
)
 
da_plot = (
    da_lat
    .sortby("z_m")
    .sel( z_m=slice( 20_000, 140_000 ) )
    .transpose(  "z_m",  "time")
)

plot_time_altitude(da_plot, da_lat = -7)