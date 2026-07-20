import xarray as xr 
import os
from scipy.signal import  butter, sosfiltfilt, detrend
import numpy as np
 

def save_results(
        u_bar, v_bar, t_bar, 
        u_prime, v_prime, t_prime
        ):

    results = xr.Dataset(
        data_vars={
            "u_bar": u_bar,
            "v_bar": v_bar,
            "t_bar": t_bar,
            "u_prime": u_prime,
            "v_prime": v_prime,
            "t_prime": t_prime,
        }
    )
    
    
    results["u_bar"].attrs = {
        "long_name": "zonal-mean zonal wind",
        "units": "m s-1",
    }
    
    results["v_bar"].attrs = {
        "long_name": "zonal-mean meridional wind",
        "units": "m s-1",
    }
    
    results["t_bar"].attrs = {
        "long_name": "zonal-mean temperature",
        "units": "K",
    }
    
    results["u_prime"].attrs = {
        "long_name": "zonal-wind perturbation",
        "units": "m s-1",
    }
    
    results["v_prime"].attrs = {
        "long_name": "meridional-wind perturbation",
        "units": "m s-1",
    }
    
    results["t_prime"].attrs = {
        "long_name": "temperature perturbation",
        "units": "K",
    }
    
  
    return results 
   


def bandpass_1d(
    x,
    low_period,
    high_period,
    fs=1.0,
    order=4,
    handle_nan=True,
    remove_trend=True
):
     

    x = np.asarray( x, dtype=np.float64 )

    output = np.full_like( x, np.nan )

    valid = np.isfinite(x)

    # Série sem dados suficientes
    if valid.sum() < 20:
        return output

    if handle_nan:
        index = np.arange(x.size)

        x_filled = np.interp(
            index,
            index[valid],
            x[valid]
        )

    else:
        if not valid.all():
            return output

        x_filled = x.copy()

    if remove_trend:
        x_filled = detrend(
            x_filled,
            type="linear"
        )

    # Frequências em ciclos por dia
    low_frequency = 1.0 / high_period
    high_frequency = 1.0 / low_period

    nyquist = 0.5 * fs

    if high_frequency >= nyquist:
        raise ValueError(
            "A frequência superior da banda deve ser "
            "menor que a frequência de Nyquist."
        )

    normalized_frequency = [
        low_frequency / nyquist,
        high_frequency / nyquist
    ]

    sos = butter(
        order,
        normalized_frequency,
        btype="bandpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        x_filled
    )

    # Restaura NaN nas posições originalmente ausentes
    filtered[~valid] = np.nan

    return filtered

def bandpass_xarray_fast(
    da,
    low_period=5,
    high_period=7,
    order=4,
    time_dim="time",
    remove_trend=True
):
    """
    Filtro Butterworth vetorizado ao longo do tempo.

    Muito mais rápido que aplicar uma função 1D
    individualmente em cada ponto espacial.
    """

    dt_days = float(
        np.median( np.diff(da[time_dim].values) / np.timedelta64(1, "D") )
    )

    fs = 1.0 / dt_days
    nyquist = fs / 2.0

    low_frequency = 1.0 / high_period
    high_frequency = 1.0 / low_period

    frequencies = [low_frequency / nyquist, high_frequency / nyquist]

    sos = butter(order, frequencies, btype="bandpass", output="sos")

    # Mantém toda a série temporal no mesmo bloco
    if da.chunks is not None:
        da = da.chunk({time_dim: -1})

    original_order = da.dims

    # Pequenas falhas são interpoladas ao longo do tempo
    valid = da.notnull()

    filled = da.interpolate_na(
        dim=time_dim,
        method="linear",
        fill_value="extrapolate"
    )

    def filter_block(values):
        """
        values pode ser multidimensional.
        O xarray coloca a dimensão core time no último eixo.
        """

        values = np.asarray( values, dtype=np.float64 )

        # if remove_trend:
        #     values = detrend(values, axis=-1, type="linear")

        return sosfiltfilt(sos, values, axis=-1)

    filtered = xr.apply_ufunc(
        filter_block, filled,
        input_core_dims = [[time_dim]],
        output_core_dims = [[time_dim]],
        vectorize = False,
        dask = "parallelized",
        output_dtypes = [np.float64],
        dask_gufunc_kwargs = {"allow_rechunk": False }
    )

    filtered = filtered.transpose(*original_order)

    # Restaura os NaN originais
    filtered = filtered.where(valid)

    filtered.attrs = da.attrs.copy()

    filtered.attrs.update({
        "filter": "Butterworth bandpass",
        "period_band":  f"{low_period}-{high_period} days",
        "filter_order": order,
        "sampling_frequency": f"{fs} samples per day",
    })

    return filtered

from dask.diagnostics import ProgressBar


def Eddy_Fluxes_by_periods(
        day_fn, 
        low_period = 5,
        high_period = 7
        ):
    infile = r"D:\database\JAWARA"
    
    files = {
        "u": os.path.join(infile, "U", f"U{day_fn}.nc"),
        "v": os.path.join(infile, "V", f"V{day_fn}.nc"),
        "t": os.path.join(infile, "T", f"T{day_fn}.nc"),
    }
    
    # Abertura com processamento por blocos
    ds_u = xr.open_dataset( files["u"])
    
    ds_v = xr.open_dataset( files["v"])
    
    ds_t = xr.open_dataset( files["t"])
    
    u = ds_u["u"]
    v = ds_v["v"]
    t = ds_t["t"]
  
    u, v, t = xr.align( u, v,  t, join="inner" )
 
    u_bar = u.mean("longitude", skipna=True)
    v_bar = v.mean("longitude", skipna=True)
    t_bar = t.mean("longitude", skipna=True)
    
    
    # # Perturbações em relação à média zonal
    
    u_prime = u - u_bar
    v_prime = v - v_bar
    t_prime = t - t_bar
    
    
    u_prime = bandpass_xarray_fast(
        u_prime,
        low_period = low_period,
        high_period = high_period,
        order=4
    )
    
    v_prime = bandpass_xarray_fast(
        v_prime,
        low_period = low_period,
        high_period = high_period ,
        order=4
    )
    
    t_prime = bandpass_xarray_fast(
        t_prime,
        low_period = low_period,
        high_period= high_period,
        order=4
    )
    
    return save_results(
            u_bar, v_bar, t_bar, 
            u_prime, v_prime, t_prime
            )

# with ProgressBar():

def main():
    day_fn = '2503'
    
    ds = Eddy_Fluxes_by_periods(day_fn)    
  
    outfile = os.path.join(
          'JAWARA/data/zonal_mean',
          f"eddy_fluxes_{day_fn}.nc"
      )
      
    ds.to_netcdf(
         outfile,
         engine="netcdf4", 
     )


main()