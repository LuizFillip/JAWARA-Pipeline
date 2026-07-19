import xarray as xr 
import os

def momentum_flux(u, v, T):
    u_bar = u.mean("longitude", skipna=True)
    v_bar = v.mean("longitude", skipna=True)
    T_bar = T.mean("longitude", skipna=True)
    
    u_prime = u - u_bar
    v_prime = v - v_bar
    T_prime = T - T_bar
    

infile = r"D:\database\JAWARA"

files = {
    "u": os.path.join(infile, "U", "U2501.nc"),
    "v": os.path.join(infile, "V", "V2501.nc"),
    "t": os.path.join(infile, "T", "T2501.nc"),
}

# Abertura com processamento por blocos
ds_u = xr.open_dataset(
    files["u"], 
)

ds_v = xr.open_dataset(
    files["v"], 
)

ds_t = xr.open_dataset(
    files["t"], 
)

u = ds_u["u"]
v = ds_v["v"]
t = ds_t["t"]



# # Garante que as três variáveis tenham as mesmas coordenadas
u, v, t = xr.align( u, v,  t, join="inner" )
# 
# # Médias zonais
u_bar = u.mean("longitude", skipna=True)
v_bar = v.mean("longitude", skipna=True)
t_bar = t.mean("longitude", skipna=True)


# # Perturbações em relação à média zonal

u_prime = u - u_bar
v_prime = v - v_bar
t_prime = t - t_bar

    
#%%%%


def save_results(
        u_bar, v_bar, t_bar, 
        u_prime, v_prime, t_prime
        ):

    results = xr.Dataset(
        data_vars={
            "u_bar": u_bar,
            "v_bar": v_bar,
            "t_bar": t_bar,
            "u_prime": u_prime,
            "v_prime": v_prime,
            "t_prime": t_prime,
        }
    )
    
    
    results["u_bar"].attrs = {
        "long_name": "zonal-mean zonal wind",
        "units": "m s-1",
    }
    
    results["v_bar"].attrs = {
        "long_name": "zonal-mean meridional wind",
        "units": "m s-1",
    }
    
    results["t_bar"].attrs = {
        "long_name": "zonal-mean temperature",
        "units": "K",
    }
    
    results["u_prime"].attrs = {
        "long_name": "zonal-wind perturbation",
        "units": "m s-1",
    }
    
    results["v_prime"].attrs = {
        "long_name": "meridional-wind perturbation",
        "units": "m s-1",
    }
    
    results["t_prime"].attrs = {
        "long_name": "temperature perturbation",
        "units": "K",
    }
    
    outfile = os.path.join(
        'JAWARA/data/zonal_mean',
        "3eddy_fluxes_25012.nc"
    )
    
    encoding = {
        variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
        for variable in results.data_vars
    }
    
    results.to_netcdf(
        outfile,
        engine="netcdf4", 
    )
    

    
save_results(
        u_bar, v_bar, t_bar, 
        u_prime, v_prime, t_prime
        )
 