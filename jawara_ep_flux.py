"""Fluxo de Eliassen-Palm para os campos zonais do JAWARA."""

from __future__ import annotations

import numpy as np
import xarray as xr
 

from scipy.signal import (
    butter,
    sosfiltfilt,
    detrend
)


def bandpass_1d(
    x,
    low_period,
    high_period,
    fs=1.0,
    order=4,
    handle_nan=True,
    remove_trend=True
):
    """
    Aplica um filtro Butterworth passa-banda a uma série 1D.

    Parameters
    ----------
    x : array
        Série temporal.

    low_period : float
        Menor período da banda, em dias.

    high_period : float
        Maior período da banda, em dias.

    fs : float
        Frequência de amostragem em amostras por dia.

    order : int
        Ordem do filtro.

    handle_nan : bool
        Interpola valores ausentes antes da filtragem.

    remove_trend : bool
        Remove média e tendência linear antes do filtro.

    Returns
    -------
    array
        Série filtrada.
    """

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

def bandpass_xarray(
    da,
    low_period=5,
    high_period=7,
    order=4,
    time_dim="time"
):
    """
    Aplica bandpass_1d ao longo da dimensão temporal
    de um xarray.DataArray.
    """

    # Intervalo temporal em dias
    dt_days = float( np.median(  
        np.diff(da[time_dim].values)
            / np.timedelta64(1, "D") ) )

    fs = 1.0 / dt_days

    print( f"Intervalo temporal: {dt_days:.3f} dia")

    print(  f"Frequência de amostragem: {fs:.1f} amostras/dia")

    # filtfilt precisa que toda a dimensão temporal
    # esteja em um único bloco Dask
    if da.chunks is not None:
        da = da.chunk({
            time_dim: -1
        })

    filtered = xr.apply_ufunc(
        bandpass_1d, da, input_core_dims = [[time_dim]],
        output_core_dims = [[time_dim]],
        kwargs={
            "low_period": low_period,
            "high_period": high_period,
            "fs": fs,
            "order": order,
            "handle_nan": True,
            "remove_trend": True,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[  np.float64 ],
        dask_gufunc_kwargs={
            "allow_rechunk": True
        }
    )

    # Restaura a ordem original
    filtered = filtered.transpose(*da.dims)

    filtered.attrs = da.attrs.copy()

    filtered.attrs.update({
        "filter": "Butterworth bandpass",
        "low_period": low_period,
        "high_period": high_period,
        "period_units": "days",
        "filter_order": order,
        "sampling_frequency": fs,
    })

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

        if remove_trend:
            values = detrend(values, axis=-1, type="linear")

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
def compute_ep_flux(
    ds: xr.Dataset,
    *,
    hour: int | None = None,
    pressure_units: str = "hPa",
    earth_radius: float = 6_371_000.0,
    omega: float = 7.2921159e-5,
    scale_height: float = 7_000.0,
    surface_density: float = 1.225,
    reference_pressure: float = 100_000.0,
    gas_constant: float = 287.05,
    specific_heat: float = 1004.0,
    min_abs_dtheta_dz: float = 1.0e-8,
    min_abs_cos_latitude: float = 1.0e-3,
) -> xr.Dataset:
    
    # required_variables = {"u_prime", "v_prime", "t_prime", "t_bar"}
    # missing = required_variables.difference(ds.data_vars)
    

    pressure = ds["level"].astype(np.float64)
    unit = pressure_units.strip().lower()
    if unit == "hpa":
        pressure_pa = pressure * 100.0
    elif unit == "pa":
        pressure_pa = pressure
    else:
        raise ValueError("pressure_units deve ser 'hPa' ou 'Pa'.")
        
    # if np.any(np.asarray(pressure_pa.values) <= 0.0):
    #     raise ValueError("Todos os níveis de pressão devem ser positivos.")

    if "altitude" in ds.coords:
        altitude = ds["altitude"].astype(np.float64)
        altitude_unit = altitude.attrs.get("units", "km").strip().lower()
        if altitude_unit in {
            "km", "kilometer", "kilometers", "quilômetro", "quilômetros"
        }:
            z_values = altitude.values * 1_000.0
        elif altitude_unit in {"m", "meter", "meters", "metro", "metros"}:
            z_values = altitude.values
        else:
            raise ValueError(
                f"Unidade de altitude não reconhecida: {altitude_unit!r}"
            )
        vertical_coordinate_source = "altitude coordinate from input Dataset"
    else:
        # Altura log-pressão: z = -H ln(p/p0).
        z_values = (
            -scale_height
            * np.log(pressure_pa.values / reference_pressure)
        )
        vertical_coordinate_source = "derived from pressure using z=-H*ln(p/p0)"

    # Uma única coordenada vertical, compartilhada por todas as variáveis.
    work = (
        ds.assign_coords(z_m=("level", z_values))
        .swap_dims({"level": "z_m"})
        .sortby("z_m")
        .sortby("latitude")
    )
    work["z_m"].attrs = {"long_name": "log-pressure altitude", "units": "m"}

    # Após swap_dims, level e pressure_pa continuam como coordenadas de z_m.
    pressure_pa = work["level"].astype(np.float64)
    if unit == "hpa":
        pressure_pa = pressure_pa * 100.0

    kappa = gas_constant / specific_heat
    theta_factor = (reference_pressure / pressure_pa) ** kappa
    theta_bar = work["t_bar"] * theta_factor
    theta_prime = work["t_prime"] * theta_factor

    uv_bar = (work["u_prime"] * work["v_prime"] ).mean(
        "longitude", skipna=True
    )
    vtheta_bar = (work["v_prime"] * theta_prime).mean(
        "longitude", skipna=True
    )

    dtheta_dz = theta_bar.differentiate("z_m", edge_order=2)
    dtheta_dz = dtheta_dz.where(np.abs(dtheta_dz) >= min_abs_dtheta_dz)

    rho0 = surface_density * np.exp(-work["z_m"] / scale_height)
    latitude_rad = np.deg2rad(work["latitude"])
    cos_latitude = np.cos(latitude_rad)
    coriolis = 2.0 * omega * np.sin(latitude_rad)
    valid_latitude = np.abs(cos_latitude) > min_abs_cos_latitude

    # Componentes do fluxo EP, ambos em kg s-2.
    f_phi = -rho0 * earth_radius * cos_latitude * uv_bar
    f_z = (
        rho0 * coriolis * earth_radius * cos_latitude * vtheta_bar / dtheta_dz
    )
    f_phi = f_phi.where(valid_latitude)
    f_z = f_z.where(valid_latitude)

    # d/dphi deve ser calculado por radiano. Como latitude está em graus,
    # convertemos a derivada retornada pelo xarray.
    d_fphi_cos_dphi = (
        (f_phi * cos_latitude).differentiate("latitude", edge_order=2)
        * 180.0 / np.pi
    )
    divergence_meridional = d_fphi_cos_dphi / (
        earth_radius * cos_latitude
    )
    divergence_vertical = f_z.differentiate("z_m", edge_order=2)
    divergence = divergence_meridional + divergence_vertical

    # Tendência zonal: m s-2; multiplicação por 86400 resulta em m s-1 day-1.
    acceleration = divergence / (
        rho0 * earth_radius * cos_latitude
    )
    acceleration_day = acceleration * 86400

    output = xr.Dataset(
        {
            "uv_bar": uv_bar,
            "theta_bar": theta_bar,
            "vtheta_bar": vtheta_bar,
            "dtheta_dz": dtheta_dz,
            "rho0": rho0,
            "F_phi": f_phi,
            "F_z": f_z,
            "divergence_meridional": divergence_meridional,
            "divergence_vertical": divergence_vertical,
            "divergence": divergence,
            "acceleration": acceleration,
            "acceleration_day": acceleration_day,
        }
    )

    metadata = {
        "uv_bar": 
            ("zonal-mean eddy momentum flux", "m2 s-2"),
        "theta_bar": 
            ("zonal-mean potential temperature", "K"),
        "vtheta_bar": 
            ("zonal-mean eddy potential-temperature flux", "K m s-1"),
        "dtheta_dz": 
            ("vertical gradient of zonal-mean potential temperature", "K m-1"),
        "rho0": 
            ("reference atmospheric density", "kg m-3"),
        "F_phi": 
            ("meridional Eliassen-Palm flux", "kg s-2"),
        "F_z": 
            ("vertical Eliassen-Palm flux", "kg s-2"),
        "divergence_meridional": 
            ("meridional EP-flux divergence term", "kg m-1 s-2"),
        "divergence_vertical": 
            ("vertical EP-flux divergence term", "kg m-1 s-2"),
        "divergence": 
            ("Eliassen-Palm flux divergence", "kg m-1 s-2"),
        "acceleration": 
            ("wave-induced zonal-wind acceleration", "m s-2"),
        "acceleration_day": 
            ("wave-induced zonal-wind tendency", "m s-1 day-1"),
    }
    for name, (long_name, units) in metadata.items():
        output[name].attrs = {"long_name": long_name, "units": units}

    output.attrs.update(
        {
            "EP_flux_formulation":
                "quasi-geostrophic, spherical log-pressure height",
            "divergence_formula": 
                "1/(a*cos(phi))*d(F_phi*cos(phi))/dphi + dF_z/dz",
            "forcing_formula": "div(F)/(rho0*a*cos(phi))",
            "vertical_coordinate_source": vertical_coordinate_source,
        }
    )

    if hour is not None:
        if not 0 <= hour <= 23:
            raise ValueError("hour deve estar entre 0 e 23.")
        output = output.where(output.time.dt.hour == hour, drop=True)

    return output


def prepare_latitude_height_field(
    da: xr.DataArray,
    *,
    time: str | np.datetime64 | None = None,
    time_mean: bool = False,
    altitude_min_km: float = 20.0,
    altitude_max_km: float = 140.0,
) -> xr.DataArray:
    """Prepara um campo latitude-altura para ``contourf``."""
    if time is not None and time_mean:
        raise ValueError("Escolha time ou time_mean, não ambos.")
    if time is not None:
        da = da.sel(time=time, method="nearest")
    elif time_mean and "time" in da.dims:
        da = da.mean("time", skipna=True)
    elif "time" in da.dims:
        raise ValueError("Informe time ou use time_mean=True.")

    return (
        da.sel(
            z_m=slice(altitude_min_km * 1_000.0, altitude_max_km * 1_000.0)
        )
        .sortby("latitude")
        .sortby("z_m")
        .transpose("z_m", "latitude")
    )


# if __name__ == "__main__":
source = "JAWARA/data/zonal_mean/eddy_fluxes_2501.nc"
results = xr.open_dataset(source)

results["u_prime"] = bandpass_xarray(
    results["u_prime"],
    low_period=5,
    high_period=7,
    order=4
)

results["v_prime"] = bandpass_xarray(
    results["v_prime"],
    low_period=5,
    high_period=7,
    order=4
)

results["t_prime"] = bandpass_xarray(
    results["t_prime"],
    low_period=5,
    high_period=7,
    order=4
)


ep = compute_ep_flux(results, hour=0)
    # ep.to_netcdf("JAWARA/data/zonal_mean/ep_flux_2501.nc")
