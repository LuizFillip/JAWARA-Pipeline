import datetime as dt 
import xarray as xr 
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

    # factor = xr.where(
    #     magnitude > magnitude_limit,
    #     magnitude_limit / magnitude,
    #     1.0
    # )

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


ep = xr.open_dataset('JAWARA/data/zonal_mean/ep_flux_2501.nc')
plot_ep_flux_vectors(
    ep,
    time = dt.datetime(2025, 1, 15),
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