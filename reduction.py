import xarray as xr 
from tqdm import tqdm 
import numpy as np 
import base as b 
import datetime as dt 
import matplotlib.pyplot as plt 
import pandas as pd 



def load_data(infile):
    ds = pd.read_csv(infile, index_col = 0)
    
    ds['altitude'] = ds['altitude'].astype(int)
 
    ds.loc[:, "time"] = pd.to_datetime(ds['time'])
    
    ds.loc[:, 'longitude'] = ((ds['longitude'] + 180) % 360) - 180
    
    return ds 

out = []
for num in tqdm(range(1, 5)):
    
    infile = f'JAWARA/data/T250{num}.txt'
    
    out.append(load_data(infile))

df = pd.concat(out)


df["time"] = pd.to_datetime(df["time"])

df = df.set_index("time") 
 

ds = df.loc[
    (df['altitude'] == 50) & 
    (df.index.time == dt.time(0, 0))
    ].copy()

mat = pd.pivot_table(
    ds,
    values = 't',
    index = 'longitude',
    columns = ds.index
    )


mat.columns = mat.columns.dayofyear


def plot(mat):
    z = mat.values
    z = b.pass_band_2d(
        mat,
        bandpass = (2.2, 13)
    )


    

import SSW25PW as p
import waves as wv


period_min, period_max = 2.1, 10



# fig, ax = plt.subplots(
#     dpi = 300,
#     nrows = 2, 
#     figsize = (12, 10))


    


start, end = 60, 90
# mat = mat.loc[:, (mat.columns >= start) &  (mat.columns <= end ) ].copy()

# filtered = b.pass_band_2d(
#     mat,
#     bandpass = (period_min, period_max)
# )

# vmax = np.nanpercentile( np.abs(filtered), 98 )

 
# # levels = np.linspace( -vmax, vmax, 30)
# img = ax[0].contourf(
#     filtered.columns, 
#     filtered.index, 
#     filtered, 
#     levels = 30, 
#     cmap = 'seismic'
#     )
 
# plt.colorbar(img)


# signal = b.bandpass_1d(
#     signal,
#     low_period= period_min,
#     high_period= period_max,
# )


# # result = p.plot_Wavelet(
# #     ax[1],
# #     time,
# #     filtered,
# #     period_min = period_min,
# #     period_max = period_max,
    
#     # )
# # wv.plot_zonalnumber_decomposition(
# #         ax[1],  
# #         filtered,
# #         period_min = period_min, 
# #         period_max = period_max + 10,
# #         y = 0.85,
# #         x = 0.05,
# #         fontsize = 30,
# #         colorbar = True
# #         )

# # ax[1].set(xlim = [-5, 5])

import spectral as sp 

import numpy as np


def inverse_wavelet_band(
    wave,
    scales,
    T_min,
    T_max,
    dj,
    dt,
    cdelta=0.776
):
    """
    Reconstrói uma banda de períodos a partir dos coeficientes
    da transformada wavelet contínua de Morlet.

    Parameters
    ----------
    wave : ndarray, shape (n_scales, n_times)
        Coeficientes complexos da CWT.

    scales : ndarray, shape (n_scales,)
        Escalas associadas aos coeficientes wavelet.

    T_min, T_max : float
        Limites inferior e superior da banda de períodos.

    dj : float
        Espaçamento logarítmico entre as escalas.

    dt : float
        Intervalo temporal da série.

    cdelta : float
        Constante de reconstrução. Para Morlet com omega0=6,
        Cdelta = 0.776.

    Returns
    -------
    recon : ndarray, shape (n_times,)
        Série temporal reconstruída na banda selecionada.
    """

    wave = np.asarray(wave)
    scales = np.asarray(scales, dtype=float)

    if wave.ndim != 2:
        raise ValueError(
            "wave deve possuir dimensões (n_scales, n_times)."
        )

    if wave.shape[0] != scales.size:
        raise ValueError(
            "O número de escalas deve coincidir com wave.shape[0]."
        )

    # Para Morlet: psi_0(0) = pi^(-1/4)
    psi0 = np.pi ** (-0.25)

    # Seleção das escalas/períodos
    mask = (scales >= T_min) & (scales <= T_max)

    if not np.any(mask):
        raise ValueError(
            "Nenhuma escala foi encontrada dentro da banda solicitada."
        )

    wave_band = wave[mask, :]
    scales_band = scales[mask]

    # sqrt(scale) com formato (n_scales_selecionadas, 1)
    sqrt_scale = np.sqrt(scales_band)[:, None]

    norm_factor = (
        dj * np.sqrt(dt)
        / (cdelta * psi0)
    )

    recon = norm_factor * np.sum(
        np.real(wave_band) / sqrt_scale,
        axis=0
    )

    return recon

def inverse_wavelet_period_band(
    wave,
    scales,
    periods,
    period_min,
    period_max,
    dj,
    dt,
    cdelta=0.776
):
    wave = np.asarray(wave)
    scales = np.asarray(scales, dtype=float)
    periods = np.asarray(periods, dtype=float)

    if wave.shape[0] != scales.size:
        raise ValueError("wave e scales possuem dimensões incompatíveis.")

    if periods.size != scales.size:
        raise ValueError("periods e scales devem possuir o mesmo tamanho.")

    psi0 = np.pi ** (-0.25)

    mask = (
        (periods >= period_min)
        & (periods <= period_max)
    )

    if not np.any(mask):
        raise ValueError(
            "Nenhum período foi encontrado na banda selecionada."
        )

    wave_band = wave[mask, :]
    scales_band = scales[mask]

    recon = (
        dj
        * np.sqrt(dt)
        / (cdelta * psi0)
        * np.sum(
            np.real(wave_band)
            / np.sqrt(scales_band)[:, None],
            axis=0
        )
    )

    return recon

#%%%%%

import pandas as pd 


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

        x_band = inverse_wavelet_period_band(
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

# ds1 = mat.loc[(mat.index == -33.7500)].T 

# time = ds1.index
# signal = ds1.values.flatten()


# wave, period, scale, coi = sp.wavelet(
#      signal, 
#      dt = 1,  
#      period_min= 2, 
#      period_max= 13
#      )

# x_q6d = inverse_wavelet_period_band(
#     wave=wave,
#     scales=scale,
#     periods=period,
#     period_min=4,
#     period_max=7,
#     dj=0.16,
#     dt=1
# ) 


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
 