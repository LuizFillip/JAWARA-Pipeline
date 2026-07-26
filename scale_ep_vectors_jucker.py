from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt 

 
lat_min_lim, lat_max_lim = -87.0, 87.0
z_min_lim, z_max_lim = 30.0, 120.0

H = 7.0         # Altura de escala (km)
a_km = 6371.0   # Raio da Terra (km)
limiar = 2.0    # Limiar de aceleração zonal para geração do gráfico
escala_viz = 15.0  # Comprimento de referência visual da seta
mag_ref_legenda = 1e5  # Valor físico de referência para a legenda

def reference_vector(
        ax, 
        X_Y,
        lat_max_lim, 
        z_max, 
        ref_max_mag
        ):
    lat_ref_pos = lat_min_lim + 8.0
    z_ref_pos = z_max_lim - 5.0

    u_ref_phys = mag_ref_legenda
    u_ref_plot = (
        u_ref_phys * (
            X_Y / ((lat_max_lim - lat_min_lim) * np.pi / 180.0)
                      )  ) * (escala_viz / ref_max_mag)
    v_ref_plot = 0.0

    ax.quiver(
        [lat_ref_pos],
        [z_ref_pos],
        [u_ref_plot],
        [v_ref_plot],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="red",
        pivot="middle",
        headwidth=3,
        headlength=4,
    )

    ax.text(
        lat_ref_pos,
        z_ref_pos + 2.5,
        f"Ref: {mag_ref_legenda:.0e}",
        color="red",
        fontsize=10,
        ha="center",
        va="bottom",
        weight="bold",
    )
    
def reference_magnitude(
        ds,
        idx_periodo, 
        idx_z, 
        stride_lat,
        stride_z, 
        dens_factor, 
        rad_lat,
        fac_x,
        fac_y
        ):
    F_phi_periodo = ds["F_phi"].isel(
        z=idx_z, time=idx_periodo
        ).transpose("lat", "z", "time").values
    F_z_periodo   = ds["F_z"].isel(
        z=idx_z, time=idx_periodo
        ).transpose("lat", "z", "time").values

  
    ref_max_mag = 0.0
    for i_t in range(len(idx_periodo)):
        f_phi_t = F_phi_periodo[::stride_lat, ::stride_z, i_t]
        f_z_t   = F_z_periodo[::stride_lat, ::stride_z, i_t]

        u_phys = f_phi_t.ravel() * np.cos(rad_lat) * dens_factor
        v_phys = f_z_t.ravel() * np.cos(rad_lat) * dens_factor * a_km

        fx = u_phys * fac_x
        fy = v_phys * fac_y

        mag_t = np.max(np.sqrt(fx**2 + fy**2))
        if mag_t > ref_max_mag:
            ref_max_mag = mag_t
            
    # print(ref_max_mag)    

    if ref_max_mag == 0.0:
        ref_max_mag = 1.0
        
    return  ref_max_mag

    
def jucker(ds, dn, delta = 5):
        

    lat = ds["lat"].values
    z_kmg = ds["z"].values / 1000.0
    times = pd.to_datetime(ds["time"].values)
    
    idx_dia = np.where(times == dn)[0][0]
   
    start = idx_dia - delta
    final = idx_dia + delta

    # Conversão para Altura Geopotencial em km
    z_km_global = (a_km * z_kmg) / (a_km - z_kmg)

    # Filtro de altitude para z >= 5 km
    idx_z = np.where(z_km_global >= z_min_lim)[0]
    z_km = z_km_global[idx_z]
 
 
    idx_periodo = np.where(
        (times >= dn - dt.timedelta(days = delta)) & 
        (times <= dn + dt.timedelta(days = delta)))[0]

    stride_lat, stride_z = 1, 2
    lat_sub = lat[::stride_lat]
    z_sub = z_km[::stride_z]

    # Malhas para os vetores
    grid_lats, grid_zs = np.meshgrid(lat_sub, z_sub, indexing="ij")
    grid_lats_flat = grid_lats.ravel()
    grid_zs_flat = grid_zs.ravel()

    rad_lat = np.deg2rad(grid_lats_flat)

    # Peso de densidade contínuo
    z_for_density = np.minimum(grid_zs_flat, z_max_lim)
    dens_factor = np.exp(z_for_density / H)

    # Estimativa da razão de aspecto geométrica (X/Y)
    aspect_ratio_est = 1.33
    fac_x = aspect_ratio_est / ((lat_max_lim - lat_min_lim) * np.pi / 180.0)
    fac_y = 1.0 / (z_max_lim - z_min_lim)

      
  
     
    ref_max_mag = reference_magnitude(
        ds, idx_periodo, idx_z,
        stride_lat, stride_z,
        dens_factor, rad_lat,
        fac_x, fac_y)
   
    
    title = f'{times[start]} - {times[final]}'
    F_phi_raw = ds["F_phi"].isel(
        z = idx_z, 
        time = slice(start, final)
        ).mean(dim="time").transpose("lat", "z").values
    
    F_z_raw   = ds["F_z"].isel(
        z = idx_z,
        time = slice(start, final)
        ).mean(dim="time").transpose("lat", "z").values
    
    div_F_raw = ds["accel"].isel(
        z = idx_z, 
        time = slice(start, final)
        ).mean(dim="time").transpose("lat", "z").values

    # minimu_f = np.min(div_F_raw)
    # maximu_f = np.max(div_F_raw)

   
    # if minimu_f <= (-limiar) or maximu_f >= limiar:
    lim = np.ceil(np.max(np.abs(div_F_raw)))
        
    # def plot_latitude_height():

    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    
    ax.set(
        xlim = [lat_min_lim, lat_max_lim],
        ylim = [z_min_lim, z_max_lim],
        xticks = np.arange(-90, 90 + 1, 10),
        yticks = np.arange(z_min_lim, z_max_lim + 1, 10),
        title = f"Fluxo EP Q6DW - {title}",
        ylabel = "Altitude (km)",
        xlabel = "Latitude (°)"
        
        )
  
  
    U_sub = F_phi_raw[::stride_lat, ::stride_z]
    V_sub = F_z_raw[::stride_lat, ::stride_z]

    U_phys = U_sub.ravel() * np.cos(rad_lat) * dens_factor
    V_phys = V_sub.ravel() * np.cos(rad_lat) * dens_factor * a_km

    # Obter a razão de aspecto geométrica real da caixa do Matplotlib
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    X_Y = bbox.width / bbox.height

    F_x = U_phys * (X_Y / ((lat_max_lim - lat_min_lim) * np.pi / 180.0))
    F_y = V_phys * (1.0 / (z_max_lim - z_min_lim))

    # Normalização pela escala de referência fixa do período
    U_plot = F_x * (escala_viz / ref_max_mag)
    V_plot = F_y * (escala_viz / ref_max_mag)

    # --- Renderização do Contorno e Aceleração ---
    levels_fill = np.linspace(-lim, lim, 31)
    levels_line = np.linspace(-lim, lim, 7)

    lat_grid_full, z_grid_full = np.meshgrid(lat, z_km, indexing="ij")

    cf = ax.contourf(
        lat_grid_full,
        z_grid_full,
        div_F_raw,
        levels=levels_fill,
        cmap= "seismic", #"PuOr_r",
        extend="both",
    )

    ax.contour(
        lat_grid_full,
        z_grid_full,
        div_F_raw,
        levels=levels_line,
        colors="black",
        linewidths=0.5,
        alpha=0.8,
    )
   
    ax.quiver(
        grid_lats_flat,
        grid_zs_flat,
        U_plot,
        V_plot,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        pivot="middle",
        headwidth=3,
        headlength=4,
    )

    ax.axvline(-7, color = 'k')
 
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Aceleração Zonal (m s⁻¹ dia⁻¹)", fontsize=12)
            
    
 
    

nc_path = 'D:\\database\\JAWARA\\no_filter\\ep_divs_q6.nc'

ds = xr.open_dataset(nc_path)
dn = dt.datetime(2025, 3, 10)
jucker(ds,  dn) 

plt.show()