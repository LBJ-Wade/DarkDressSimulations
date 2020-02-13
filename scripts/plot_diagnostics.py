from math import sqrt
from os import listdir
from os.path import isfile, join

import click
import matplotlib.pyplot as plt
import numpy as np
from amuse.io import read_set_from_file
from amuse.units import units


def parse_file(path):
    bodies = read_set_from_file(path, "hdf5")

    imbh = None
    bh = None
    for b in bodies:
        if imbh is None and b.name == "IMBH":
            imbh = b
        elif bh is None and b.name == "BH":
            bh = b

    if imbh is None or bh is None:
        raise ValueError("IMBH and/or BH not found!")

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
    snap_names = [f for f in listdir(snap_dir) if isfile(join(snap_dir, f))]
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
