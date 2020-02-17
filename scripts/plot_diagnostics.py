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
from utils import get_bhs


def parse_file(path):
    bodies = read_set_from_file(path, "hdf5")
    imbh, bh = get_bhs(bodies)[0]

    t = imbh.time.in_(units.s).number
    z_imbh = imbh.position[2].in_(units.parsec).number
    z_bh = bh.position[2].in_(units.parsec).number
    bh_separation = sqrt(
        (
            (
                imbh.position.in_(units.parsec).number
                - bh.position.in_(units.parsec).number
            )
            ** 2
        ).sum()
    )
    r_com_bh = (
        imbh.mass.number * imbh.position.in_(units.parsec).number
        + bh.mass.number * bh.position.in_(units.parsec).number
    ) / (imbh.mass.number + bh.mass.number)

    r_com = bodies.center_of_mass().number
    K = bodies.kinetic_energy().in_(units.MSun * units.kms ** 2).number
    U = bodies.potential_energy().in_(units.MSun * units.kms ** 2).number

    return t, z_imbh, z_bh, bh_separation, r_com_bh, r_com, K, U


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


def plot(ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us):
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
    fig.savefig("diagostics.png")


@click.command()
@click.option(
    "--snap_dir",
    default="../snapshots",
    help="directory containing simulation snapshots",
)
@click.option("--snap_format", default="hdf5")
def run(snap_dir, snap_format):
    # Get number of snaps and prefix
    snap_names = [
        f for f in listdir(snap_dir) if isfile(join(snap_dir, f)) and f[0] != "."
    ]
    print(snap_names)
    n_snaps = len(snap_names)
    snap_prefix = snap_names[0].split("_")[0]

    # Parse snapshots
    ts = []
    z_imbhs = []
    z_bhs = []
    bh_separations = []
    r_com_bhs = []
    r_coms = []
    Ks = []
    Us = []

    print(f"> Loading {n_snaps} snapshots")

    for i in range(n_snaps):
        snap_path = join(snap_dir, f"{snap_prefix}_{i}.{snap_format}")
        print(f"> {snap_path}")
        t, z_imbh, z_bh, bh_separation, r_com_bh, r_com, K, U = parse_file(snap_path)
        ts.append(t)
        z_imbhs.append(z_imbh)
        z_bhs.append(z_bh)
        bh_separations.append(bh_separation)
        r_com_bhs.append(r_com_bh)
        r_coms.append(r_com)
        Ks.append(K)
        Us.append(U)
    ts = np.array(ts)
    z_imbhs = np.array(z_imbhs)
    z_bhs = np.array(z_bhs)
    bh_separations = np.array(bh_separations)
    r_com_bhs = np.array(r_com_bhs)
    r_coms = np.array(r_coms)
    Ks = np.array(Ks)
    Us = np.array(Us)

    print(f"Parsed {n_snaps} snapshots")

    plot(ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us)
    print("Generated diagostic plots")


if __name__ == "__main__":
    run()
