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
    
    
# results = xr.open_dataset('JAWARA/data/zonal_mean/eddy_fluxes_2501.nc')

# results["uv_bar"] = ( 
#     results["u_prime"]  * results["v_prime"] 
#     ).mean(
#             "longitude",
#             skipna=True
#         )

# results["uv_bar"].attrs = {
#     "long_name": "zonal-mean meridional eddy momentum flux",
#     "units": "m2 s-2",
# }


# results = results.assign_coords(
#     z_m=(  "level", results["altitude"].values * 1000.0 )
# )

# results["z_m"].attrs = {
#     "long_name": "altitude",
#     "units": "m",
# }


# results = add_scale_heght(results)


# results = add_density(results)
# results = Zonal_Heat_flux(results)
 

# source = "JAWARA/data/zonal_mean/eddy_fluxes_2501.nc"
# ds = xr.open_dataset(source)

# ds.isel(level = 0)