"""Fluxo de Eliassen-Palm para os campos zonais do JAWARA."""

from __future__ import annotations

import numpy as np
import xarray as xr


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
    
    required_variables = {"u_prime", "v_prime", "t_prime", "t_bar"}
    # missing = required_variables.difference(ds.data_vars)
    

    pressure = ds["level"].astype(np.float64)
    unit = pressure_units.strip().lower()
    if unit == "hpa":
        pressure_pa = pressure * 100.0
    elif unit == "pa":
        pressure_pa = pressure
    else:
        raise ValueError("pressure_units deve ser 'hPa' ou 'Pa'.")
    if np.any(np.asarray(pressure_pa.values) <= 0.0):
        raise ValueError("Todos os níveis de pressão devem ser positivos.")

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

    uv_bar = (work["u_prime"] * work["v_prime"]).mean(
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
        rho0
        * coriolis
        * earth_radius
        * cos_latitude
        * vtheta_bar
        / dtheta_dz
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
        "uv_bar": ("zonal-mean eddy momentum flux", "m2 s-2"),
        "theta_bar": ("zonal-mean potential temperature", "K"),
        "vtheta_bar": ("zonal-mean eddy potential-temperature flux", "K m s-1"),
        "dtheta_dz": ("vertical gradient of zonal-mean potential temperature", "K m-1"),
        "rho0": ("reference atmospheric density", "kg m-3"),
        "F_phi": ("meridional Eliassen-Palm flux", "kg s-2"),
        "F_z": ("vertical Eliassen-Palm flux", "kg s-2"),
        "divergence_meridional": ("meridional EP-flux divergence term", "kg m-1 s-2"),
        "divergence_vertical": ("vertical EP-flux divergence term", "kg m-1 s-2"),
        "divergence": ("Eliassen-Palm flux divergence", "kg m-1 s-2"),
        "acceleration": ("wave-induced zonal-wind acceleration", "m s-2"),
        "acceleration_day": ("wave-induced zonal-wind tendency", "m s-1 day-1"),
    }
    for name, (long_name, units) in metadata.items():
        output[name].attrs = {"long_name": long_name, "units": units}

    output.attrs.update(
        {
            "EP_flux_formulation": "quasi-geostrophic, spherical log-pressure height",
            "divergence_formula": "1/(a*cos(phi))*d(F_phi*cos(phi))/dphi + dF_z/dz",
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
data = xr.open_dataset(source)
ep = compute_ep_flux(data, hour=0)
    # ep.to_netcdf("JAWARA/data/zonal_mean/ep_flux_2501.nc")

#%%%%
F_plot = ep.isel(time = 10) 

F_plot["z_m"] /1000
F_plot = F_plot.where(F_plot.z_m > 20e3)
import matplotlib.pyplot as plt 


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm


def plot_ep_flux_vectors(
    ep,
    time,
    latitude_min=-87,
    latitude_max=10,
    altitude_min=15,
    altitude_max=90,
    latitude_step=4,
    altitude_step=4,
    vector_scale=14,
    vertical_exaggeration=1.0,
    percentile=98,
    figsize=(13, 8)
):
    """
    Plota os vetores do fluxo EP sobre a tendência do vento zonal.

    Parameters
    ----------
    ep : xarray.Dataset
        Deve conter F_phi, F_z e acceleration_day.

    time : str ou datetime
        Data utilizada no gráfico.

    latitude_min, latitude_max : float
        Intervalo latitudinal.

    altitude_min, altitude_max : float
        Intervalo vertical em km.

    latitude_step, altitude_step : int
        Subamostragem dos vetores.

    vector_scale : float
        Escala gráfica do quiver. Valores maiores produzem
        setas menores.

    vertical_exaggeration : float
        Amplificação gráfica da componente vertical.

    percentile : float
        Percentil usado para limitar a escala de cores.
    """

    # Seleção temporal
    field = ep.sel(
        time=time,
        method="nearest"
    )

    # Seleção espacial
    field = (
        field
        .sortby("latitude")
        .sortby("z_m")
        .sel(
            latitude=slice(
                latitude_min,
                latitude_max
            ),
            z_m=slice(
                altitude_min * 1000,
                altitude_max * 1000
            )
        )
    )

    latitude = field["latitude"]
    altitude_km = field["z_m"] / 1000.0

    acceleration = field[ "acceleration_day"]

    # Escala simétrica do campo de fundo
    color_limit = float(
        np.nanpercentile(
            np.abs(acceleration.values),
            percentile
        )
    )

    norm = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0,
        vmax=color_limit
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # Campo de aceleração
    contour = ax.contourf(
        latitude,
        altitude_km,
        acceleration,
        levels=31,
        cmap="RdBu_r",
        norm=norm,
        extend="both"
    )

    # Contornos adicionais
    contour_lines = ax.contour(
        latitude,
        altitude_km,
        acceleration,
        levels=np.linspace(
            -color_limit,
            color_limit,
            9
        ),
        colors="0.25",
        linewidths=0.6
    )

    ax.clabel(
        contour_lines,
        inline=True,
        fontsize=8,
        fmt="%.1f"
    )

    # Subamostragem dos vetores
    vectors = field.isel(
        latitude=slice(
            None,
            None,
            latitude_step
        ),
        z_m=slice(
            None,
            None,
            altitude_step
        )
    )

    F_phi = vectors["F_phi"]
    F_z = vectors["F_z"]

    # --------------------------------------------------
    # Normalização gráfica
    # --------------------------------------------------

    # Normaliza cada componente por uma escala robusta
    scale_phi = float(
        np.nanpercentile(
            np.abs(F_phi.values),
            95
        )
    )

    scale_z = float(
        np.nanpercentile(
            np.abs(F_z.values),
            95
        )
    )

    if scale_phi == 0:
        scale_phi = 1.0

    if scale_z == 0:
        scale_z = 1.0

    U = F_phi / scale_phi

    V = ( vertical_exaggeration * F_z  / scale_z
    ) 
    # Limita vetores extremos
    magnitude = np.sqrt(
        U**2 + V**2
    )

    magnitude_limit = float(
        np.nanpercentile(
            magnitude.values,
            95
        )
    )

    factor = xr.where(
        magnitude > magnitude_limit,
        magnitude_limit / magnitude,
        1.0
    )

    # U = U * factor
    # V = V * factor

    # Máscara para valores inválidos
    valid = (
        np.isfinite(U)
        & np.isfinite(V)
    )

    U = U.where(valid)
    V = V.where(valid)

    quiver = ax.quiver(
        vectors["latitude"],
        vectors["z_m"] / 1000.0,
        U,
        V,
        color="black",
        angles="uv",
        scale_units="inches",
        scale=vector_scale,
        width=0.003,
        headwidth=4,
        headlength=5,
        headaxislength=4.5,
        pivot="middle"
    )

    # Chave gráfica
    ax.quiverkey(
        quiver,
        X=0.84,
        Y=1.04,
        U=1,
        label="EP flux vector",
        labelpos="E",
        coordinates="axes"
    )

    # Barra de cores
    colorbar = fig.colorbar(
        contour,
        ax=ax,
        pad=0.02,
        aspect=30
    )

    colorbar.set_label(
        r"Aceleração zonal "
        r"(m s$^{-1}$ dia$^{-1}$)"
    )

    selected_time = np.datetime_as_string(
        field["time"].values,
        unit="D"
    )

    ax.set_title(
        f"Fluxo EP – {selected_time}",
        fontweight="bold"
    )

    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Altitude (km)")

    ax.set_xlim(
        latitude_min,
        latitude_max
    )

    ax.set_ylim(
        altitude_min,
        altitude_max
    )

    ax.tick_params(
        direction="out"
    )

    plt.tight_layout()

    return fig, ax

import datetime as dt 

plot_ep_flux_vectors(
    ep,
    time = dt.datetime(2025, 1, 1),
    latitude_min=-80,
    latitude_max=80,
    altitude_min=15,
    altitude_max=140,
    latitude_step=4,
    altitude_step=4,
    vector_scale=14,
    vertical_exaggeration=1.0,
    percentile=98,
    figsize=(13, 8)
)