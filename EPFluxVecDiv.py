"""Cálculo do fluxo de Eliassen-Palm e de sua divergência.

Tradução revisada de ``EPFluxVecDiv.jl`` para Python. Os campos de entrada
devem estar organizados como (latitude, nível, tempo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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

        dz =  Rd * T_mean / g  * np.log(p[k - 1] / p[k])

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

@dataclass(frozen=True)
class EPConstants:
    """Constantes físicas e parâmetros da coordenada vertical."""

    earth_radius: float = 6_371_000.0  # m
    omega: float = 7.2921159e-5  # rad s-1
    scale_height: float = 7_000.0  # m; ajuste ao valor de CoefsRecon.jl
    reference_pressure: float = 100_000.0  # Pa
    surface_density: float = 1.225  # kg m-3


def pressure_to_height(
    pressure: np.ndarray,
    *,
    scale_height: float,
    reference_pressure: float,
) -> np.ndarray:
    """Converte pressão (Pa) em altura log-pressão (m)."""
    pressure = np.asarray(pressure, dtype=np.float64)
    return -scale_height * np.log(pressure / reference_pressure)


def reference_density(
    height_m: np.ndarray, *, surface_density: float, scale_height: float
) -> np.ndarray:
    """Retorna rho0(z) = rho_s exp(-z/H)."""
    return surface_density * np.exp(-np.asarray(height_m) / scale_height)


def gradient_1d_2nd_order(values: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Derivada de segunda ordem, inclusive nas bordas, em grade não uniforme."""
    values = np.asarray(values, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
  
    return np.gradient(values, coordinate, edge_order=2)


def _gradient_axis(
        values: np.ndarray,
        coordinate: np.ndarray,
        axis: int) -> np.ndarray:
    """Aplica a derivada de segunda ordem ao longo de um eixo."""
    return np.apply_along_axis(
        gradient_1d_2nd_order, axis, values, np.asarray(coordinate)
    )


def calculate_ep_flux_3d(
    latitude_deg: np.ndarray,
    height_m: np.ndarray,
    rho0: np.ndarray,
    uv_eddy: np.ndarray,
    vtheta_eddy: np.ndarray,
    theta_bar: np.ndarray,
    *,
    earth_radius: float = 6_371_000.0,
    omega: float = 7.2921159e-5,
    min_abs_dtheta_dz: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula F_phi, F_z e a tendência zonal induzida pelas ondas.

    Parâmetros
    ----------
    uv_eddy
        Covariância zonal ``u'v'`` em m2 s-2.
    vtheta_eddy
        Covariância zonal ``v'theta'`` em K m s-1.
    theta_bar
        Temperatura potencial média zonal em K.

    Retorna
    -------
    F_phi, F_z, acceleration
        Arrays (latitude, nível, tempo). A aceleração está em m s-2.
    """
    latitude_deg = np.asarray(latitude_deg, dtype=np.float64)
    height_m = np.asarray(height_m, dtype=np.float64)
    rho0 = np.asarray(rho0, dtype=np.float64)
    uv_eddy = np.asarray(uv_eddy, dtype=np.float64)
    vtheta_eddy = np.asarray(vtheta_eddy, dtype=np.float64)
    theta_bar = np.asarray(theta_bar, dtype=np.float64)

    expected = (latitude_deg.size, height_m.size)
    
    if uv_eddy.shape != vtheta_eddy.shape or uv_eddy.shape != theta_bar.shape:
        raise ValueError("uv_eddy, vtheta_eddy e theta_bar devem ter a mesma forma.")
    if uv_eddy.ndim != 3 or uv_eddy.shape[:2] != expected:
        raise ValueError("Os campos devem ter dimensões (latitude, nível, tempo).")
    if rho0.shape != (height_m.size,):
        raise ValueError("rho0 deve possuir um valor para cada nível vertical.")

    phi = np.deg2rad(latitude_deg)
    cos_phi = np.cos(phi)
    coriolis = 2.0 * omega * np.sin(phi)

    # Broadcasting: latitude x nível x tempo.
    metric = earth_radius * cos_phi[:, None, None]
    density = rho0[None, :, None]
    f3d = coriolis[:, None, None]

    dtheta_dz = _gradient_axis(theta_bar, height_m, axis=1)
    stable = np.abs(dtheta_dz) >= min_abs_dtheta_dz
    ratio = np.divide(
        vtheta_eddy,
        dtheta_dz,
        out=np.full_like(vtheta_eddy, np.nan),
        where=stable,
    )

    f_phi = -density * metric * uv_eddy
    f_z = density * f3d * metric * ratio

    # div(F) = [1/(a cos(phi))] d[F_phi cos(phi)]/dphi + dF_z/dz.
    weighted_f_phi = f_phi * cos_phi[:, None, None]
    d_weighted_dphi = _gradient_axis(weighted_f_phi, phi, axis=0)
    d_fz_dz = _gradient_axis(f_z, height_m, axis=1)

    valid_lat = np.abs(cos_phi) > 1.0e-2
    term_phi = np.divide(
        d_weighted_dphi,
        metric,
        out=np.full_like(d_weighted_dphi, np.nan),
        where=valid_lat[:, None, None],
    )
    divergence = term_phi + d_fz_dz
    acceleration = np.divide(
        divergence,
        density * metric,
        out=np.full_like(divergence, np.nan),
        where=valid_lat[:, None, None],
    )
    return f_phi, f_z, acceleration


def process_case(
    flux_file: str | Path,
    theta_file: str | Path,
    output_file: str | Path,
    pressure_levels_pa: np.ndarray,
    *,
    constants: EPConstants = EPConstants(),
) -> None:
    """Processa um par de arquivos reproduzindo os recortes do script Julia."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("Instale xarray e netCDF4 para ler arquivos NetCDF.") from exc
    height_m = pressure_to_height(
        pressure_levels_pa,
        scale_height=constants.scale_height,
        reference_pressure=constants.reference_pressure,
    )
    rho0 = reference_density(
        height_m,
        surface_density=constants.surface_density,
        scale_height=constants.scale_height,
    )
    nlev = height_m.size

    with xr.open_dataset(flux_file) as flux, xr.open_dataset(theta_file) as mean:
        # Equivalente a [:, 2:end, :] e [:, 2:nlev+1, :] do Julia.
        uv = flux["uv_bar"].isel({flux["uv_bar"].dims[1]: slice(1, None)}).values
        vtheta = flux["vθ_bar"].isel({flux["vθ_bar"].dims[1]: slice(1, None)}).values
        theta = mean["theta_bar"].isel(
            {mean["theta_bar"].dims[1]: slice(1, nlev + 1)}
        ).values
        latitude = flux["lat"].values.astype(np.float64)
        time = flux["time"].values

    if uv.shape[1] != nlev or vtheta.shape[1] != nlev:
        raise ValueError(
            "O recorte vertical dos fluxos não coincide com pressure_levels_pa."
        )

    f_phi, f_z, accel = calculate_ep_flux_3d(
        latitude,
        height_m,
        rho0,
        uv,
        vtheta,
        theta,
        earth_radius=constants.earth_radius,
        omega=constants.omega,
    )
    save_ep_flux_data(output_file, latitude, height_m, time, f_phi, f_z, accel)


