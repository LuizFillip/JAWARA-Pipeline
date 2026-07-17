# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:18:00 2026

@author: Luiz
"""

import base as b 

def fixed_spatial_boxes_xr(
    ds,
    lat_step=10,
    lon_step=10,
    lat_center=None,
    lat_name="latitude",
    lon_name="longitude",
    drop=True,
):
   

    ds = ds.copy()

    lat = ds[lat_name]
    lon = ds[lon_name]

    # Longitude no intervalo 0–360°
    lon360 = lon % 360

    ds = ds.assign_coords(
        lon360=(lon_name, lon360.data)
    )

    # ---------------------------------------------------------
    # Caixas de latitude
    # ---------------------------------------------------------
    if lat_center is None:

        lat_box = (
            np.floor((lat + 90) / lat_step)
            * lat_step
            - 90
            + lat_step / 2
        )

        ds = ds.assign_coords(
            lat_box=(lat_name, lat_box.data)
        )

    else:

        half = lat_step / 2

        mask = (
            (lat >= lat_center - half)
            & (lat < lat_center + half)
        )

        ds = ds.where(mask, drop=drop)

        lat_box = xr.full_like(
            ds[lat_name],
            fill_value=float(lat_center),
            dtype=float,
        )

        ds = ds.assign_coords(
            lat_box=(lat_name, lat_box.data)
        )

    # ---------------------------------------------------------
    # Caixas de longitude
    # ---------------------------------------------------------
    lon360 = ds["lon360"]

    lon_box = (
        np.floor(lon360 / lon_step)
        * lon_step
        + lon_step / 2
    ) % 360

    # Versão para plot no intervalo −180° a 180°
    lon_box_plot = (
        (lon_box + 180) % 360
    ) - 180

    ds = ds.assign_coords(
        lon_box=(lon_name, lon_box.data),
        lon_box_plot=(lon_name, lon_box_plot.data),
    )

    return ds


def plot_longitude_time_by_altitude(
    ds,
    variable="v",
    altitudes=(20, 30, 40, 50, 60, 70, 80, 90, 100),
    lat_center=-7,
    lat_width=10,
    doy_min=None,
    doy_max=None,
    remove_time_mean=True,
    cmap="seismic",
    levels=21,
    figsize=(15, 11),
):
    """
    Plota flutuações longitude-tempo para diferentes altitudes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset com dimensões time, level, latitude e longitude.
    variable : str
        Variável que será plotada.
    altitudes : sequence
        Altitudes desejadas, em km.
    lat_center : float
        Latitude central da faixa analisada.
    lat_width : float
        Largura total da faixa latitudinal, em graus.
    doy_min, doy_max : float or None
        Intervalo de dias do ano.
    remove_time_mean : bool
        Remove a média temporal em cada longitude e altitude.
    cmap : str
        Mapa de cores.
    levels : int
        Número de níveis do contourf.
    figsize : tuple
        Tamanho da figura.
    """

    ds = add_log_pressure_height(ds)

    lat_min = lat_center - lat_width / 2
    lat_max = lat_center + lat_width / 2

    # Funciona tanto para latitude crescente quanto decrescente
    if ds.latitude[0] > ds.latitude[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    da = (
        ds[variable]
        .sel(latitude=lat_slice)
        .mean("latitude", skipna=True)
    )

    # Cria coordenada DOY
    doy = (
        da.time.dt.dayofyear
        + da.time.dt.hour / 24
        + da.time.dt.minute / 1440
    )

    da = da.assign_coords(doy=("time", doy.values))

    # if doy_min is not None:
    #     da = da.where(da.doy >= doy_min, drop=True)

    # if doy_max is not None:
    #     da = da.where(da.doy <= doy_max, drop=True)

    # Ordena altitude para permitir interpolação
    da = da.sortby("altitude")

    # Interpola verticalmente para as altitudes solicitadas
    da_alt = da.interp(
        altitude=np.asarray(altitudes, dtype=float)
    )

    # if remove_time_mean:
    #     # Flutuação relativa à média temporal de cada longitude
    #     da_alt = da_alt - da_alt.mean(
    #         dim="time",
    #         skipna=True
    #     )

    n_alt = len(altitudes)
    ncols = 3
    nrows = int(np.ceil(n_alt / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        dpi=300, 
    )

    axes = np.atleast_1d(axes).ravel()

    panel_labels = [
        f"({chr(97 + i)})"
        for i in range(n_alt)
    ]

    for i, (ax, altitude) in enumerate(
        zip(axes, altitudes)
    ):
        field = da_alt.sel(altitude=altitude)
        
        raw = field.transpose( "longitude", "time" ) 
        
        z = b.pass_band_2d(
            raw,
            bandpass= (2.2, 13), 
            pandas = False  
        )

        vmax = np.nanpercentile( np.abs(z), 98 )

        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        contour_levels = np.linspace( -vmax, vmax, levels  )

        im = ax.contourf(
            field.doy,
            field.longitude,
            z,
            levels = contour_levels,
            cmap = cmap,
            extend = "both",
        )

        ax.set_title(
            f"{panel_labels[i]} {altitude:.0f} km",
            fontsize=16,
        )

        ax.set_ylim(-180, 180)
        ax.set_yticks( np.arange(-180, 181, 60) )

        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.025,
            fraction=0.045,
        )
  
    for ax in axes[n_alt:]:
        ax.remove()

    # Rótulos somente nas bordas
    for i, ax in enumerate(axes[:n_alt]):
        row = i // ncols
        col = i % ncols

        if col == 0:
            ax.set_ylabel("Longitude (°)")

        if row == nrows - 1:
            ax.set_xlabel("Day of Year")

    fig.suptitle(
        f"{variable} fluctuations "
        rf"(center latitude = {lat_center:.0f}°)",
        fontsize=21,
    )

    return fig, axes, da_alt