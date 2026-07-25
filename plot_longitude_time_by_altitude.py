import base as b
import JAWARA as jw
import numpy as np
import matplotlib.pyplot as plt


ds = jw.concat_datasets(
    "U",
    latitude=-7,
    zonal_mean=False
)
 

#%%%


def plot_longitude_time_by_altitude(
    ds,
    variable="u",
    altitudes=(20, 30, 40, 50, 60, 70, 80, 90, 100),
    lat_center=-7,
    doy_min=None,
    doy_max=None,
    remove_time_mean=True,
    cmap="seismic",
    levels=21,
    figsize=(15, 11),
):

    # Adiciona a coordenada de altitude
    ds_alt = jw.add_log_pressure_height(ds)

    # Seleciona a variável
    da = ds_alt[variable]

    # Troca level por altitude
    da = (
        da
        .swap_dims({"level": "altitude"})
        .sortby("altitude")
        .transpose("altitude", "longitude", "time")
    )

    # Longitude: 0–360 para -180–180
    longitude = ((da.longitude + 180) % 360) - 180

    da = (
        da
        .assign_coords(longitude=longitude)
        .sortby("longitude")
    )

    # Média diária:
    # pressupõe dados regulares de 6 em 6 horas
    da = (
        da .coarsen(
            time=4,
            boundary="trim"
        )
        .mean()
    )

    # Cria DOY após calcular a média diária
    doy = (
        da.time.dt.dayofyear
        + da.time.dt.hour / 24
        + da.time.dt.minute / 1440
        + da.time.dt.second / 86400
    )

    da = da.assign_coords(
        doy=("time", doy.values)
    )

    # Recorte temporal
    if doy_min is not None:
        da = da.where( da.doy >= doy_min, drop=True )

    if doy_max is not None:
        da = da.where( da.doy <= doy_max, drop=True  )

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

    panel_labels = [  f"({chr(97 + i)})"  for i in range(n_alt) ]

    for i, (ax, altitude) in enumerate(zip(axes, altitudes)):

        field = da.sel(altitude = altitude, method = "nearest")

        raw = field.transpose( "longitude", "time" )

        # # Remove a média temporal em cada longitude
        # if remove_time_mean:
        #    z = raw - raw.mean(dim = "longitude",  skipna=True )

        z = b.pass_band_2d(
            raw,
            bandpass=(2.2, 13),
            pandas=False
        )
 
        vmax = np.nanpercentile( np.abs(z), 98 )
 

        contour_levels = np.linspace( -vmax,  vmax, levels )

        im = ax.contourf(
            field.doy,
            field.longitude,
            z,
            levels=contour_levels,
            cmap=cmap,
            extend="both"
        )

        selected_altitude = float(
            field.altitude.values
        )
        ax.axvline(
            54, lw = 4, 
            color = 'green', 
            linestyle = '--', 
            label = 'SSW onset')
        ax.set(title  = 
            f"{panel_labels[i]} "
            f"{selected_altitude:.0f} km", 
            ylim = [-180, 180],
            yticks = np.arange(-180, 181, 60)
        )
 
        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            pad=0.025,
            fraction=0.045,
        )

    # Remove painéis extras
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
        rf"{variable} fluctuations "
        rf"at {lat_center:.0f}° latitude",
        fontsize=21,
    )

    fig.subplots_adjust(
        top=0.93,
        hspace=0.22,
        wspace=0.28
    )

    return fig, axes, da

alts = np.arange(20, 110, 10)

fig, axes, da_daily = plot_longitude_time_by_altitude(
    ds,
    variable="u",
    altitudes=alts,
    lat_center=-7,
    doy_min=1,
    doy_max=120,
    remove_time_mean=True,
    cmap="seismic",
    levels=21,
    figsize=(15, 11),
)

plt.show()