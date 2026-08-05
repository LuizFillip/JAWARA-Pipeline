import xarray as xr 
import JAWARA as jw
import pandas as pd 
import os 
import warnings
from tqdm import tqdm 
import numpy as np 
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="unique with argument that is not.*",
    category=FutureWarning
)



def load_data_netcdf(fn):

    ds = xr.open_dataset(fn)
    
    ds = jw.add_log_pressure_height(ds)
    
     
    doy = (
        ds.time.dt.dayofyear
        + ds.time.dt.hour / 24
        + ds.time.dt.minute / 1440
    )
    
    ds = ds.assign_coords(doy=("time", doy.values))
    
    return ds 



def get_parameters_by_latitude(da, latitude = -7):
    ds_sel = da.sel(latitude = latitude, method = "nearest")
    
    sel_lon = ds_sel['t'].to_pandas().to_frame('t')
    
    sel_lon['altitude'] = da['altitude'].item()
    
    sel_lon["time"] = pd.Timestamp(da["time"].values)
    sel_lon = sel_lon.reset_index()
    
    return sel_lon 

def get_parameters_by_longitude(da, longitude = -35):
    ds_sel = da.sel(longitude = longitude, method = "nearest")
    
    sel_lon = ds_sel['t'].to_pandas().to_frame('t')
    
    sel_lon['altitude'] = da['altitude'].item()
    
    sel_lon["time"] = pd.Timestamp(da["time"].values)
    sel_lon = sel_lon.reset_index()
    
    return sel_lon 


def run_by_levels(ds, lat = True):
    out = []
    for level in ds.level.values:
        
        df = ds.sel(level = level)
        
        if lat:
            out.append(
                get_parameters_by_latitude( df )
                )
        else:
            out.append(
                get_parameters_by_longitude( df )
                )
    
    return pd.concat(out)


def run_by_time(ds, desc = ''):
    out = []
    for time in tqdm(ds.time.values, desc):
     
        out.append(run_by_levels(ds.sel(time = time)))
        
        
    return pd.concat(out)


def main():
    
    infile = "D:\\database\\JAWARA\\T\\"
    save_in = 'JAWARA/data/'
    
    for fn in os.listdir(infile):
        ds =  run_by_time(load_data_netcdf(infile + fn), fn)
        
        ds.to_csv(save_in + fn.replace('nc', 'txt'))
        

 
def concat_datasets(io, latitude=60.0, zonal_mean=True):
    """
    Carrega e concatena arquivos mensais JAWARA de janeiro a abril de 2025.

    Parameters
    ----------
    io : str
        Nome da variável e da pasta, por exemplo: "U", "V" ou "T".
    latitude : float, default=60.0
        Latitude selecionada pelo método do vizinho mais próximo.
    zonal_mean : bool, default=True
        Se True, calcula a média zonal sobre longitude.

    Returns
    -------
    xr.Dataset ou xr.DataArray
        Dados concatenados, ordenados temporalmente e sem tempos duplicados.
    """
    datasets = []

    base_path = Path(r"D:\database\JAWARA") / io

    for month in range(1, 5):
        filename = base_path / f"{io}25{month:02d}.nc"

        if not filename.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {filename}"
            )

        ds_month = load_data_netcdf(filename)

        ds_month = ds_month.sel(
            latitude=latitude,
            method="nearest",
        )

        if zonal_mean and "longitude" in ds_month.dims:
            ds_month = ds_month.mean(
                dim="longitude",
                skipna=True,
                keep_attrs=True,
            )

        datasets.append(ds_month)

    ds = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    # Ordena cronologicamente
    ds = ds.sortby("time")

    # Remove tempos duplicados, preservando a primeira ocorrência
    time_values = np.asarray(ds["time"].values)

    _, unique_indices = np.unique(
        time_values,
        return_index=True,
    )

    ds = ds.isel(
        time=np.sort(unique_indices)
    )

    return ds