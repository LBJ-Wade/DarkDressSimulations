from os import listdir
from math import sqrt
from os.path import isfile, join
from amuse.io import read_set_from_file
import numpy as np
from amuse.units import units


def get_bhs(bodies):
    """
    Finds the IMBH and BH particles.
    """
    imbh = None
    i_imbh = None
    bh = None
    i_bh = None
    for i, b in enumerate(bodies):
        if imbh is None and b.name == "IMBH":
            imbh = b
            i_imbh = i
        elif bh is None and b.name == "BH":
            bh = b
            i_bh = i

    if imbh is None or bh is None:
        raise ValueError("IMBH and/or BH not found!")

    return (imbh, bh), (i_imbh, i_bh)


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


def parse_files(snap_dir, snap_format, verbose):
    # Get number of snaps and prefix
    snap_names = [
        f for f in listdir(snap_dir) if isfile(join(snap_dir, f)) and f[0] != "."
    ]
    if verbose:
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

    if verbose:
        print(f"> Loading {n_snaps} snapshots")

    for i in range(n_snaps):
        snap_path = join(snap_dir, f"{snap_prefix}_{i}.{snap_format}")
        if verbose:
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

    return ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us
