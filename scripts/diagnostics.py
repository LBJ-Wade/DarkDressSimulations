from math import sqrt
from os import listdir
from os.path import isfile, join

import click
import matplotlib.pyplot as plt

# BJK
# import matplotlib
# matplotlib.use('MacOSX')

import numpy as np
from amuse.io import read_set_from_file
from amuse.units import units
from utils import get_bhs, parse_files


# BK: This function should ultimately return the r bins and
# density profile, but for now it just plots rho(r).
def plot_density_profile(path):
    bodies = read_set_from_file(path, "hdf5")

    M_IMBH = None
    # Read in IMBH mass and remove non-DM particles
    for b in bodies:
        if b.name == "IMBH":
            M_IMBH = b.mass.in_(units.MSun).number
        if b.name != "dmp":
            bodies = bodies - b

    if M_IMBH is None:
        raise ValueError("IMBH not found!")

    # Bin the radial positions of the DM particles
    N_DM = len(bodies)
    r_vals = np.array(
        [(bodies[i].position.length()).in_(units.parsec).number for i in range(N_DM)]
    )

    N_bins = 50
    r_bins = np.geomspace(1e-11, 1e-5, N_bins + 1)

    r_counts, r_bins = np.histogram(r_vals, r_bins)
    r_counts = (
        r_counts * bodies[0].mass.in_(units.MSun).number
    )  # Multiply by mass per particle
    r_c = np.sqrt(r_bins[:-1] * r_bins[1:])  # Bin centres

    # Volume per r-bin
    dV = (4 * np.pi / 3) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)

    # Underlying density profile
    rho_sp = 226  # M_sun/pc^3
    gamma = 7.0 / 3.0
    r_sp = ((3 - gamma) * (0.2 ** (3.0 - gamma)) * M_IMBH / (2 * np.pi * rho_sp)) ** (
        1.0 / 3.0
    )
    r_t = 1e-5 * r_sp * (100 / M_IMBH) ** (3 / 2)
    alpha = 2.0
    rho_IC = rho_sp * (r_c / r_sp) ** (-gamma) / ((1 + r_c / r_t) ** alpha)

    plt.figure()

    plt.loglog(r_c, rho_IC, linestyle="--")
    plt.loglog(r_c, r_counts / dV)

    plt.xlabel(r"$r$ [pc]")
    plt.ylabel(r"$\rho_\mathrm{DM}$ [$M_\odot$ pc$^{-3}$]")

    plt.show()


def plot(ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us, fig_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    ax = axes[0, 0]
    ax.plot(ts, z_imbhs, label="IMBH")
    ax.plot(ts, z_bhs, label="BH")
    ax.set_ylabel(r"$z$ (pc)")

    ax = axes[0, 1]
    ax.plot(ts, bh_separations)
    ax.set_ylabel(r"Black hole separation (pc)")

    ax = axes[1, 0]
    ax.plot(ts, r_com_bhs, label="BHs")
    ax.plot(ts, r_coms, label="Total")
    ax.set_ylabel(r"COM coordinates (pc)")

    ax = axes[1, 1]
    ax.plot(ts, Ks + Us, label=r"$E_{\mathrm{tot}}$")
    ax.set_ylabel(r"Energy ($M_\odot \mathrm{km}^2/\mathrm{s}^2$)")

    for ax in axes.flatten():
        ax.set_xlabel(r"$t$ (s)")
        ax.legend()

    fig.tight_layout()
    fig.savefig(fig_path)


@click.command()
@click.option(
    "--snap_dir",
    default="../snapshots",
    help="directory containing simulation snapshots",
)
@click.option("--snap_format", default="hdf5")
@click.option("--fig_path", default="diagnostics.png", help="path for saving figure")
def run(snap_dir, snap_format, fig_path):
    ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us = parse_files(
        snap_dir, snap_format, verbose=True
    )
    print(f"> Parsed {len(ts)} snapshots")

    plot(ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us, fig_path)
    print("> Generated diagnostic plots")


if __name__ == "__main__":
    run()
