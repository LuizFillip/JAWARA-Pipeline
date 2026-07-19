

def plot_heigth_latitude(F_plot):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    
    vmax = float(
        np.nanpercentile(
            np.abs(F_plot.values),
            90
        )
    )
    
    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )
    F_plot = F_plot.where(
        F_plot["altitude"] > 20,
        drop=True
    )
    
    contour = ax.contourf(
        F_plot["latitude"],
        F_plot["altitude"],
        F_plot,
        levels=61,
        cmap="RdBu_r",
        # norm=norm,
        # extend="both"
    )
    
    cbar = fig.colorbar(
        contour,
        ax=ax,
        pad=0.02
    )
    
    cbar.set_label(
        r"$F_{\phi}$ (kg s$^{-2}$)"
    )
    
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(
        r"Horizontal EP flux $F_{\phi}$ — January 2025"
    )
    
    # ax.set_ylim(20, 110)
    # ax.set_xlim(-90, 90)
    
    plt.tight_layout()
    plt.show()
