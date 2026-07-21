 

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import xarray as xr


def scale_ep_vectors_jucker(
    F_phi,
    F_z,
    rho0,
    ax,
    latitude_name="latitude",
    vertical_name="z_m",
    earth_radius=6_371_000.0,
    percentile=95,
    include_2pi_over_g= True,
    gravity=9.80665
):
    """
    Escalona vetores EP conforme Jucker (2021)
    para um gráfico latitude × altura.

    Parameters
    ----------
    F_phi, F_z : xarray.DataArray
        Componentes físico-dinâmicos do fluxo EP.

    rho0 : xarray.DataArray
        Densidade atmosférica de referência em kg m-3.

    ax : matplotlib.axes.Axes
        Eixo já criado, com limites espaciais definidos.

    percentile : float
        Percentil usado apenas para definir um comprimento
        gráfico global. Não altera a direção dos vetores.

    include_2pi_over_g : bool
        Inclui o fator comum 2π/g da Equação 18 de
        Jucker (2021).

    Returns
    -------
    U, V : xarray.DataArray
        Componentes escalonados para uso em ax.quiver().
    """

    F_phi, F_z, rho0 = xr.align(
        F_phi,
        F_z,
        rho0,
        join="exact"
    )

    latitude = F_phi[latitude_name]
    cos_latitude = np.cos(
        np.deg2rad(latitude)
    )

    # Equação 18 de Jucker (2021)
    F_phi_hat = (
        earth_radius**2
        * cos_latitude
        * F_phi
    )

    F_z_hat = (
        earth_radius
        * cos_latitude
        * F_z
        / rho0
    )

    if include_2pi_over_g:

        common_factor = ( 2.0 * np.pi / gravity )

        F_phi_hat = ( common_factor * F_phi_hat  )

        F_z_hat = (
            common_factor * F_z_hat
        )

    # Limites físicos representados
    latitude_min, latitude_max = (
        ax.get_xlim()
    )

    z_min_km, z_max_km = (
        ax.get_ylim()
    )

    delta_phi_rad = np.deg2rad(
        abs(
            latitude_max
            - latitude_min
        )
    )

    # z deve ser usado em metros
    delta_z_m = (
        abs(z_max_km - z_min_km)
        * 1000.0
    )

    if delta_phi_rad == 0:
        raise ValueError(
            "O intervalo de latitude é nulo."
        )

    if delta_z_m == 0:
        raise ValueError(
            "O intervalo de altitude é nulo."
        )

    # Dimensões reais do eixo na figura
    ax.figure.canvas.draw()

    bbox = (
        ax.get_window_extent()
        .transformed(
            ax.figure
            .dpi_scale_trans
            .inverted()
        )
    )

    width_inches = bbox.width
    height_inches = bbox.height

    # Equações 6–9 e Tabela 2 de Jucker
    alpha = (
        width_inches
        / delta_phi_rad
    )

    beta = (
        height_inches
        / delta_z_m
    )

    U = alpha * F_phi_hat
    V = beta * F_z_hat

    # Apenas um fator global pode ser aplicado.
    # Não normalizar os componentes separadamente.
    magnitude = np.sqrt(
        U**2 + V**2
    )

    global_scale = float(
        np.nanpercentile(
            magnitude.values,
            percentile
        )
    )

    if (
        not np.isfinite(global_scale)
        or global_scale == 0
    ):
        global_scale = 1.0

    U = U / global_scale
    V = V / global_scale

    U.name = "F_phi_scaled"
    V.name = "F_z_scaled"

    U.attrs = {
        "long_name": (
            "Jucker-scaled meridional EP-flux component"
        ),
        "scaling": "Jucker (2021)",
    }

    V.attrs = {
        "long_name": (
            "Jucker-scaled vertical EP-flux component"
        ),
        "scaling": "Jucker (2021)",
    }

    return U, V
 

def scale_ep_vectors_pressure(
    F_phi_z,
    F_z,
    rho0,
    latitude_name="latitude",
    earth_radius=6_371_000.0,
    gravity=9.80665,
    latitude_scale=np.pi,
    pressure_scale=100_000.0,
    vertical_axis="altitude",
    percentile=95
):
    """
    Escalona os vetores EP segundo a formulação:

        F_phi_tilde =
            cos(phi) * (F_phi_p/a) / pi

        F_p_tilde =
            cos(phi) * F_p / 1e5

    Os campos de entrada estão em coordenadas de altura:

        F_phi_z = rho0 * F_phi_p
        F_p = -g * F_z

    Parameters
    ----------
    F_phi_z : DataArray
        Componente meridional calculada em coordenada
        de altura, em kg s-2.

    F_z : DataArray
        Componente vertical em coordenada de altura,
        em kg s-2.

    rho0 : DataArray
        Densidade de referência, em kg m-3.

    vertical_axis : {"altitude", "pressure"}
        Coordenada vertical usada no gráfico.

    percentile : float ou None
        Percentil usado para um único fator gráfico global.
        Use None para não aplicar normalização adicional.
    """

    F_phi_z, F_z, rho0 = xr.align(
        F_phi_z,
        F_z,
        rho0,
        join="exact"
    )

    latitude = F_phi_z[latitude_name]

    cos_latitude = np.cos(
        np.deg2rad(latitude)
    )

    # -----------------------------------------
    # Conversão altura -> pressão
    # -----------------------------------------

    F_phi_pressure = (  F_phi_z / rho0  )

    F_p = (  -gravity * F_z )

    # -----------------------------------------
    # Escalonamento da figura
    # -----------------------------------------

    U = (
        cos_latitude
        * F_phi_pressure
        / earth_radius
        / latitude_scale
    )

    V_pressure = (
        cos_latitude
        * F_p
        / pressure_scale
    )

    # Em um gráfico de pressão, Fp positivo aponta para
    # pressões maiores. Em um gráfico de altitude, o eixo
    # vertical tem sentido oposto.
    if vertical_axis == "altitude":

        V = -V_pressure

    elif vertical_axis == "pressure":

        V = V_pressure

    else:

        raise ValueError(
            "vertical_axis deve ser "
            "'altitude' ou 'pressure'."
        )

    # Um único fator adicional pode ser utilizado
    # sem modificar as direções relativas
    if percentile is not None:

        magnitude = np.sqrt(
            U**2 + V**2
        )

        global_scale = float(
            np.nanpercentile(
                magnitude.values,
                percentile
            )
        )

        if (
            np.isfinite(global_scale)
            and global_scale > 0
        ):
            U = U / global_scale
            V = V / global_scale

    U.name = "F_phi_scaled"
    V.name = "F_vertical_scaled"

    U.attrs = {
        "long_name": (
            "scaled meridional EP-flux component"
        ),
        "scaling": (
            "cos(phi)*(F_phi_pressure/a)/pi"
        ),
    }

    V.attrs = {
        "long_name": (
            "scaled vertical EP-flux component"
        ),
        "scaling": (
            "cos(phi)*(-F_p)/1e5 for altitude plot"
        ),
    }

    return U, V

ep = xr.open_dataset('JAWARA/data/zonal_mean/ep_flux_2502.nc')

date = "2025-02-18"

field = (
    ep
    .sel(
        time=date,
        method="nearest"
    )
    .sortby("latitude")
    .sortby("z_m")
    .sel(  latitude=slice(0, 90),
           z_m=slice(15_000, 120_000)
    )
)


acceleration = field[
    "acceleration_day"
]

limit = float(
    np.nanpercentile(
        np.abs(acceleration.values),
        98
    )
)

norm = TwoSlopeNorm(
    vmin=-limit,
    vcenter=0,
    vmax=limit
)

fig, ax = plt.subplots(
    figsize=(13, 8)
)

contour = ax.contourf(
    field["latitude"],
    field["z_m"] / 1000.0,
    acceleration,
    levels=31,
    cmap="RdBu_r",
    norm=norm,
    extend="both"
)

# ax.set_xlim(0, -90)
ax.set_ylim(15, 110)

ax.set_xlabel("Latitude (°)")
ax.set_ylabel("Altitude (km)")

vectors = field.isel(
    latitude=slice(None, None, 2),
    z_m=slice(None, None, 2)
)

U, V = scale_ep_vectors_pressure(
    F_phi_z=vectors["F_phi"],
    F_z=vectors["F_z"],
    rho0=vectors["rho0"],
    vertical_axis="altitude",
    percentile=95
)

U = U.transpose(
    "z_m",
    "latitude"
)

V = V.transpose(
    "z_m",
    "latitude"
)

# quiver = ax.quiver(
#     vectors["latitude"],
#     vectors["z_m"] / 1000.0,
#     U,
#     V,
#     color="black",
#     angles="uv",
#     scale_units="inches",
#     scale=2.0,
#     width=0.003,
#     headwidth=4,
#     headlength=5,
#     headaxislength=4.5,
#     pivot="middle"
# )

q = ax.quiver(
    vectors.latitude,
    vectors.z_m / 1000,
    U,
    V,
    angles="uv",
    scale_units="inches",
    scale=0.8,
    color="black",
    width=0.003,
    pivot="middle"
)


colorbar = fig.colorbar(
    contour,
    ax=ax,
    pad=0.02
)

colorbar.set_label(
    r"Aceleração zonal "
    r"(m s$^{-1}$ day$^{-1}$)"
)

print(
    "U:",
    float(np.nanpercentile(
        np.abs(U),
        50
    )),
    float(np.nanpercentile(
        np.abs(U),
        95
    ))
)

print(
    "V:",
    float(np.nanpercentile(
        np.abs(V),
        50
    )),
    float(np.nanpercentile(
        np.abs(V),
        95
    ))
)

ratio = (
    np.abs(V)
    / np.abs(U)
)

print(
    "Mediana |V/U|:",
    float(
        np.nanmedian(ratio)
    )
)