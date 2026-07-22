import xarray as xr 
import JAWARA as jw
import pandas as pd 
import os 
import warnings
from tqdm import tqdm 

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
        ds =  run_by_time(load_data(infile + fn), fn)
        
        ds.to_csv(save_in + fn.replace('nc', 'txt'))
        

