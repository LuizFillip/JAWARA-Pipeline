import xarray as xr 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

a = 6_371_000.0


def add_scale_heght(results):
    H = 7.0       # altura de escala, km
    p0 = 1000.0   # pressão de referência, hPa
    
    altitude = -H * np.log(
        results["level"] / p0
    )
    
    results = results.assign_coords(
        altitude=("level", altitude.values)
    )
    
    results["altitude"].attrs = {
        "long_name": "log-pressure altitude",
        "units": "km",
    }
    
    return results 


def density(altitude):    
    rho_sfc = 1.225  # kg m-3
    H_m = 7000.0     # m
    
    z_m = altitude * 1000.0
    
    return rho_sfc * np.exp( -z_m / H_m )


def Zonal_Heat_flux(results):
 
    latitude_rad = np.deg2rad(results["latitude"])
    
    cos_lat = np.cos(latitude_rad)
    
    F_phi = -results["rho0"] * a * cos_lat * results["uv_bar"] 
    
    F_phi.name = "F_phi"
    
    F_phi.attrs = {
        "long_name": "meridional component of Eliassen-Palm flux",
        "units": "kg s-2",
    }
  
    results["F_phi"] = F_phi
    
    return results 

def add_density(results):
    rho0 = density(results['altitude'])
    rho0.name = "rho0"
    rho0.attrs = {
        "long_name": "reference atmospheric density",
        "units": "kg m-3",
    }
    
    results["rho0"] = rho0
    
    return results
    
    
results = xr.open_dataset('JAWARA/data/zonal_mean/eddy_fluxes_2501.nc')

results["uv_bar"] = ( 
    results["u_prime"]  * results["v_prime"] 
    ).mean(
            "longitude",
            skipna=True
        )

results["uv_bar"].attrs = {
    "long_name": "zonal-mean meridional eddy momentum flux",
    "units": "m2 s-2",
}


results = results.assign_coords(
    z_m=(  "level", results["altitude"].values * 1000.0 )
)

results["z_m"].attrs = {
    "long_name": "altitude",
    "units": "m",
}


results = add_scale_heght(results)


results = add_density(results)
results = Zonal_Heat_flux(results)
 
def zonal_acceleration(results):

    F_phi = results["F_phi"].sortby("latitude")
 
    cos_lat = np.cos(np.deg2rad(F_phi["latitude"]))
    
    # Derivada inicialmente por grau
    derivative_degree = (F_phi * cos_lat).differentiate("latitude")
    
    # Converte derivada por grau para derivada por radiano
    derivative_radian = (derivative_degree * 180 / np.pi)
    
    div_phi = ( derivative_radian / (a * cos_lat) )
    
    rho0 = density(results['altitude'])
    
    accel_phi = div_phi / (rho0 * a * cos_lat)
    
    #m s-2 -> m s-1 day-1
    accel_phi_day = accel_phi * 86400
    
    accel_phi_day.name = "accel_phi"
    
    accel_phi_day.attrs = {
        "long_name": (
            "meridional contribution to EP-flux divergence"
        ),
        "units": "m s-1 day-1",
    }
    
    return accel_phi


# def plot_diff_time():
#     dates = pd.date_range(
#         '2025-01-01', '2025-01-31', freq = '3D')
    
        
#     fig, ax = plt.subplots(
#         ncols = 3, 
#         nrows = 3,
#         figsize = (16, 12)
#         )
        
        
        
#     for ax, time in zip(ax.flat, dates):
#         F_plot = df.sel(time = time)
         
#         contour = ax.contourf(
#             F_plot["latitude"],
#             F_plot["altitude"],
#             F_plot,
#             levels=61,
#             cmap='seismic',  
#         )
        
#         cbar = fig.colorbar(
#             contour,
#             ax = ax,
#             pad = 0.02
#         )
            
#     fig, ax = plt.subplots(
#         # ncols = 3, 
#         # nrows = 3,
#         # figsize = (16, 12)
#         )
#     F_plot = df.isel(time = 1)
#     contour = plt.contourf(
#          F_plot["latitude"],
#          F_plot["altitude"],
#          F_plot,
#          levels=61,
#          cmap='seismic',  
#      )
     
#     cbar = fig.colorbar(
#          contour,
#          ax = ax,
#          pad = 0.02
#      )

results = results.where(results.time.dt.hour == 0, drop=True )



F_phi = zonal_acceleration(results)



  


def calculate_vertical_ep_flux(
    ds,
    v_prime=None,
    t_prime=None,
    t_bar=None,
    altitude_name="altitude",
    level_name="level",
    latitude_name="latitude",
    longitude_name="longitude",
    pressure_units="hPa",
    earth_radius=6_371_000.0,
    omega=7.2921159e-5,
    scale_height=7000.0,
    surface_density=1.225,
    reference_pressure=100_000.0,
    gas_constant=287.05,
    specific_heat=1004.0,
    min_dtheta_dz=1e-8
):
     

    if v_prime is None:
        v_prime = ds["v_prime"]

    if t_prime is None:
        t_prime = ds["t_prime"]

    if t_bar is None:
        t_bar = ds["t_bar"]

    if altitude_name not in ds.coords:
        raise ValueError(
            f"A coordenada {altitude_name!r} não foi encontrada."
        )

    # Alinha as variáveis
    v_prime, t_prime, t_bar = xr.align(
        v_prime,
        t_prime,
        t_bar,
        join="inner"
    )

    # Pressão
    pressure = ds[level_name].astype(np.float64)

    if pressure_units.lower() == "hpa":
        pressure_pa = pressure * 100.0

    elif pressure_units.lower() == "pa":
        pressure_pa = pressure

    else:
        raise ValueError(
            "pressure_units deve ser 'hPa' ou 'Pa'."
        )

    pressure_pa.name = "pressure"
    pressure_pa.attrs["units"] = "Pa"

    # Temperatura potencial
    kappa = gas_constant / specific_heat

    potential_temperature_factor = (
        reference_pressure / pressure_pa
    ) ** kappa

    theta_bar = (
        t_bar
        * potential_temperature_factor
    )

    theta_prime = (
        t_prime
        * potential_temperature_factor
    )

    theta_bar.name = "theta_bar"
    theta_bar.attrs = {
        "long_name": "zonal-mean potential temperature",
        "units": "K",
    }

    theta_prime.name = "theta_prime"
    theta_prime.attrs = {
        "long_name": "potential-temperature perturbation",
        "units": "K",
    }

    # Fluxo meridional de temperatura potencial
    vtheta_bar = ( v_prime * theta_prime ).mean( longitude_name,
        skipna=True
    )

    vtheta_bar.name = "vtheta_bar"
    vtheta_bar.attrs = {
        "long_name": (
            "zonal-mean meridional eddy "
            "potential-temperature flux"
        ),
        "units": "K m s-1",
    }

    # Altitude em metros
    altitude = ds[altitude_name].astype(
        np.float64
    )

    altitude_units = (
        altitude.attrs
        .get("units", "km")
        .lower()
    )

    if altitude_units in ["km", "kilometer", "kilometers"]:
        z_m = altitude * 1000.0
    else:
        z_m = altitude

    z_m.name = "z_m"
    z_m.attrs = {
        "long_name": "log-pressure altitude",
        "units": "m",
    }

    # Substitui level por z_m para calcular a derivada
    theta_bar_z = (
        theta_bar
        .assign_coords(
            z_m=(
                level_name,
                z_m.values
            )
        )
        .swap_dims({
            level_name: "z_m"
        })
        .sortby("z_m")
    )

    vtheta_bar_z = (
        vtheta_bar
        .assign_coords(
            z_m=(
                level_name,
                z_m.values
            )
        )
        .swap_dims({
            level_name: "z_m"
        })
        .sortby("z_m")
    )

    # Gradiente vertical da temperatura potencial média
    dtheta_dz = theta_bar_z.differentiate(
        "z_m",
        edge_order=2
    )

    dtheta_dz = dtheta_dz.where(
        np.abs(dtheta_dz) >= min_dtheta_dz
    )

    dtheta_dz.name = "dtheta_dz"
    dtheta_dz.attrs = {
        "long_name": (
            "vertical gradient of zonal-mean "
            "potential temperature"
        ),
        "units": "K m-1",
    }

    # Densidade atmosférica de referência
    rho0 = (
        surface_density
        * np.exp(
            -theta_bar_z["z_m"]
            / scale_height
        )
    )

    rho0.name = "rho0"
    rho0.attrs = {
        "long_name": "reference atmospheric density",
        "units": "kg m-3",
    }

    # Latitude e parâmetro de Coriolis
    latitude = theta_bar_z[latitude_name]

    latitude_rad = np.deg2rad(latitude)

    coriolis = ( 2.0 * omega * np.sin(latitude_rad))

    cos_latitude = np.cos(latitude_rad)

    # Componente vertical do fluxo EP
    F_z = ( rho0 * coriolis * earth_radius * cos_latitude * vtheta_bar_z / dtheta_dz
    )

    F_z.name = "F_z"

    F_z.attrs = {
        "long_name": (
            "vertical component of "
            "Eliassen-Palm flux"
        ),
        "units": "kg s-2",
        "formula": (
            "rho0*f*a*cos(latitude)"
            "*mean(v_prime*theta_prime)"
            "/dtheta_bar_dz"
        ),
    }

    return xr.Dataset(
        {
            "theta_bar": theta_bar_z,
            "vtheta_bar": vtheta_bar_z,
            "dtheta_dz": dtheta_dz,
            "rho0": rho0,
            "F_z": F_z,
        }
    )
 
Fz = calculate_vertical_ep_flux(
    results,
    v_prime=None,
    t_prime=None,
    t_bar=None,
    altitude_name="altitude",
    level_name="level",
    latitude_name="latitude",
    longitude_name="longitude",
    pressure_units="hPa",
    earth_radius=6_371_000.0,
    omega=7.2921159e-5,
    scale_height=7000.0,
    surface_density=1.225,
    reference_pressure=100_000.0,
    gas_constant=287.05,
    specific_heat=1004.0,
    min_dtheta_dz=1e-8
)
 
F_phi = results["F_phi"].sortby("latitude")

F_phi_z = (
    F_phi
    .assign_coords(
        z_m = ("level", results["altitude"].values * 1000.0)
    )
    .swap_dims({ "level": "z_m" })
    .sortby("z_m")
    .sortby("latitude")
)


F_z = ( Fz.sortby("z_m") .sortby("latitude"))

F_phi_z, F_z = xr.align( F_phi_z, F_z, join="inner" )
 

def calculate_ep_flux_divergence(
    F_phi,
    F_z,
    latitude_name="latitude",
    vertical_name="z_m",
    earth_radius=6_371_000.0,
    polar_limit=1e-3
):
    """
    Calcula a divergência do fluxo de Eliassen-Palm:

        div(F) =
            1 / (a cos(phi))
            * d(F_phi cos(phi))/dphi
            + dF_z/dz

    Parameters
    ----------
    F_phi : xarray.DataArray
        Componente meridional do fluxo EP, em kg s-2.

    F_z : xarray.DataArray
        Componente vertical do fluxo EP, em kg s-2.

    latitude_name : str
        Nome da coordenada de latitude.

    vertical_name : str
        Nome da coordenada vertical em metros.

    earth_radius : float
        Raio da Terra em metros.

    polar_limit : float
        Limite mínimo de |cos(phi)| para evitar
        singularidades próximas aos polos.

    Returns
    -------
    xarray.Dataset
        Contém os termos meridional e vertical e a
        divergência total.
    """

    F_phi, F_z = xr.align(
        F_phi,
        F_z,
        join="inner"
    )

    F_phi = (
        F_phi
        .sortby(latitude_name)
        .sortby(vertical_name)
    )

    F_z = (
        F_z
        .sortby(latitude_name)
        .sortby(vertical_name)
    )

    latitude = F_phi[latitude_name]

    phi_rad = np.deg2rad(latitude)
    cos_phi = np.cos(phi_rad)

    # Evita a singularidade nos polos
    valid_latitude = (
        np.abs(cos_phi) > polar_limit
    )

    # -------------------------------------------
    # Termo meridional
    # -------------------------------------------

    weighted_F_phi = (
        F_phi * cos_phi
    )

    # xarray calcula inicialmente a derivada por grau
    derivative_per_degree = (
        weighted_F_phi
        .differentiate(
            latitude_name,
            edge_order=2
        )
    )

    # Conversão:
    # d/dphi(rad) = (180/pi) d/dlatitude(degree)
    derivative_per_radian = (
        derivative_per_degree
        * 180.0
        / np.pi
    )

    divergence_meridional = (
        derivative_per_radian
        / (
            earth_radius
            * cos_phi
        )
    ).where(valid_latitude)

    divergence_meridional.name = (
        "divergence_meridional"
    )

    divergence_meridional.attrs = {
        "long_name": (
            "meridional contribution to "
            "EP-flux divergence"
        ),
        "units": "kg m-1 s-2",
    }

    # -------------------------------------------
    # Termo vertical
    # -------------------------------------------

    divergence_vertical = (
        F_z
        .differentiate(
            vertical_name,
            edge_order=2
        )
    )

    divergence_vertical.name = (
        "divergence_vertical"
    )

    divergence_vertical.attrs = {
        "long_name": (
            "vertical contribution to "
            "EP-flux divergence"
        ),
        "units": "kg m-1 s-2",
    }

    # -------------------------------------------
    # Divergência total
    # -------------------------------------------

    divergence_total = (
        divergence_meridional
        + divergence_vertical
    )

    divergence_total.name = (
        "EP_flux_divergence"
    )

    divergence_total.attrs = {
        "long_name": (
            "Eliassen-Palm flux divergence"
        ),
        "units": "kg m-1 s-2",
        "formula": (
            "1/(a*cos(phi))*"
            "d(F_phi*cos(phi))/dphi + dF_z/dz"
        ),
    }

    return xr.Dataset(
        {
            "divergence_meridional":
                divergence_meridional,

            "divergence_vertical":
                divergence_vertical,

            "divergence_total":
                divergence_total,
        }
    )
        
ep = calculate_ep_flux_divergence(
    F_phi,
    F_z,
    latitude_name="latitude",
    vertical_name="z_m",
    earth_radius=6_371_000.0,
    polar_limit=1e-3
)

ep 

 