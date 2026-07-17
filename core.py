import xarray as xr 
import numpy as np
import pandas as pd 
import os 
import warnings
from tqdm import tqdm 

warnings.filterwarnings(
    "ignore",
    message="unique with argument that is not.*",
    category=FutureWarning
)

Rd = 287.05
g = 9.80665


def pressure_to_height_hypsometric(
    temperature,
    pressure_hpa,
    reference_height=0.0
):
    """
    temperature:
        DataArray com dimensão 'level', em K.

    pressure_hpa:
        DataArray ou vetor de pressão em hPa.

    Retorna:
        altitude em metros.
    """

    p = np.asarray(pressure_hpa, dtype=float) * 100.0
    T = np.asarray(temperature, dtype=float)

    z = np.full_like(T, np.nan, dtype=float)
    z[0] = reference_height

    for k in range(1, len(p)):
        T_mean = 0.5 * (T[k - 1] + T[k])

        dz = (
            Rd * T_mean / g
            * np.log(p[k - 1] / p[k])
        )

        z[k] = z[k - 1] + dz

    return z



def add_log_pressure_height(
    ds,
    level_name="level",
    p0=1000.0,
    scale_height=7.0,
):
    """
    Adiciona ao Dataset a coordenada de altura log-pressão:

        z = -H ln(p/p0)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset com uma coordenada vertical de pressão.
    level_name : str
        Nome da coordenada vertical.
    p0 : float
        Pressão de referência, em hPa.
    scale_height : float
        Altura de escala, em km.

    Returns
    -------
    xr.Dataset
        Dataset com a coordenada 'altitude'.
    """

    pressure = ds[level_name].astype(float)

    altitude = -scale_height * np.log(pressure / p0)

    return ds.assign_coords(
        altitude=(level_name, altitude.values)
    )



 



 
def load_data(fn):

    ds = xr.open_dataset(fn)
    
    ds = add_log_pressure_height(ds)
    
     
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

def run_by_levels(ds):
    out = []
    for level in ds.level.values:
     
        out.append(
            get_parameters_by_latitude(
                ds.sel(level = level)
                )
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
        ds =  run_by_time(load_data(infile + fn), fn)
        
        ds.to_csv(save_in + fn.replace('nc', 'txt'))