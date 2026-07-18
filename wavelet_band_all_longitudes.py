# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 07:20:17 2026

@author: Luiz
"""

import spectral as sp 



def wavelet_band_all_longitudes(
    mat,
    period_min=4,
    period_max=7,
    wavelet_period_min=2,
    wavelet_period_max=13,
    dt=1.0,
    dj=0.16
):
     
    reconstructed = pd.DataFrame(
        index=mat.index,
        columns=mat.columns,
        dtype=float
    )

    for lon in mat.index:

        signal = mat.loc[lon].to_numpy(dtype=float)

        # Interpola valores ausentes apenas ao longo do tempo
        if np.isnan(signal).any():

            signal = (
                pd.Series(signal, index=mat.columns)
                .interpolate(
                    method="linear",
                    limit_direction="both"
                )
                .to_numpy()
            )

        # Remove a média temporal de cada longitude
        signal_anom = signal - np.mean(signal)

        wave, period, scale, coi = sp.wavelet(
            signal_anom,
            dt=dt,
            period_min=wavelet_period_min,
            period_max=wavelet_period_max
        )

        x_band = sp.inverse_wavelet_period_band(
            wave=wave,
            scales=scale,
            periods=period,
            period_min=period_min,
            period_max=period_max,
            dj=dj,
            dt=dt
        )

        reconstructed.loc[lon, :] = x_band

    return reconstructed
 

q6d = wavelet_band_all_longitudes(
    mat,
    period_min=4,
    period_max=7,
    wavelet_period_min=1,
    wavelet_period_max=15,
    dt=1,
    dj=0.16
)


plt.contourf(
    q6d.columns, 
    q6d.index, 
    q6d.values, 
    levels = 30, 
    cmap = 'seismic'
    )
 