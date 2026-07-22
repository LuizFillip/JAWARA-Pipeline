import xarray as xr
import JAWARA as jw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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
    
    cbar = fig.colorbar(img, ax=ax, pad = 0.1)
     
    ax.set(
        xlabel="Time",
        ylabel="Altitude (km)",
        ylim=(15, 120),
        
    )

def avg_stratosphere(ax, field):
    ax_right = ax.twinx()
    
    series_32km = field.sel(altitude = 32,  method="nearest" )
    
    ax_right.plot(
        series_32km.time,
        series_32km,
        color="black",
        linewidth=2,
        label=(
            f"{float(series_32km.altitude):.1f} km"
        )
    )
    
    ax_right.set_ylabel(
        "Zonal wind at 32 km (m/s)"
    )
    
    
def plot_field(ax, da, io):
    info = desc[io]
    
    field = da.sel( altitude = slice(15, 120) )
    field = da[io.lower()]
    
    field = (
        field.transpose("level", "time").swap_dims(
            {"level": "altitude"}
            ).sortby("altitude")
        )
     
        
    plot_time_height_diagram(ax, info, field)
    
    avg_stratosphere(ax, field)
 



desc = {
    'V': {'label': "Meridional wind", 'unit': '(m/s)', 'cmap': "RdBu_r"},
    'U': {'label': "Zonal wind", 'unit': '(m/s)', 'cmap': "RdBu_r"},
    'T': {'label': "Temperature", 'unit': '(K)', 'cmap': 'turbo'}
    }
fig, ax = plt.subplots(
    figsize = (14, 10),
    nrows = 2,
    dpi = 300,
)

latitude = 60

da_zon = concat_datasets('U')
da_tem = concat_datasets('T')

  
  
plot_field(ax[0], da_tem, io = 'T')
plot_field(ax[1], da_zon, io = 'U')

 
plt.show()