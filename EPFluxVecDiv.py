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

 


 
