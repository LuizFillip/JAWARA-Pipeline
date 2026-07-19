# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:29:41 2026

@author: Luiz
"""

def vertical_coordinate():
    # Organiza a coordenada vertical em metros
    F_z_z = (
        results["F_z"]
        .assign_coords(
            z_m=(
                "level",
                results["altitude"].values * 1000.0
            )
        )
        .swap_dims({
            "level": "z_m"
        })
        .sortby("z_m")
    )
    
    # Derivada vertical
    dFz_dz = F_z_z.differentiate(  "z_m" )
    
    # O termo horizontal também deve usar a mesma dimensão vertical
    div_phi_z = (
        div_phi
        .assign_coords(
            z_m=(
                "level",
                results["altitude"].values * 1000.0
            )
        )
        .swap_dims({
            "level": "z_m"
        })
        .sortby("z_m")
    )
    
    rho0_z = (
        rho0
        .assign_coords(
            z_m=(
                "level",
                results["altitude"].values * 1000.0
            )
        )
        .swap_dims({
            "level": "z_m"
        })
        .sortby("z_m")
    )
    
    divergence = div_phi_z + dFz_dz
    
    accel = (
        divergence
        / (
            rho0_z
            * a
            * cos_lat
        )
    )
    
    accel_day = accel * 86_400.0
    
    accel_day.name = "EP_flux_divergence"
    
    accel_day.attrs = {
        "long_name": (
            "normalized Eliassen-Palm flux divergence"
        ),
        "units": "m s-1 day-1",
    }
    
    
