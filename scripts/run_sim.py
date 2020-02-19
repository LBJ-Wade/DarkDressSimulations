import warnings
from math import sqrt

from os.path import join
import click
import matplotlib.pyplot as plt
import numpy as np
from amuse.datamodel import Particles
from amuse.io import read_set_from_file, write_set_to_file
from amuse.support.exceptions import AmuseWarning
from amuse.units import nbody_system, units
from tqdm import tqdm

from utils import get_bhs, parse_files
from diagnostics import plot as plot_diagnostics

# Ignore citation warnings
warnings.filterwarnings("ignore", category=AmuseWarning)

TEST_SNAP_DIR = "../test_snapshots"
TEST_FIG_DIR = "../test_figures"
# This is the accuracy we require for the solver in terms of the relative
# change in BH separation per orbit.
# For a system with mass ratio 10^5 embedded in a static DM halo with initial
# separation 10^-8 pc, the minimum change in separation per cycle is ~10^-15
# pc. To measure changes on this order to ~0.5%, the separation of two BHs with
# no dress in a circular orbit must be stable to one part in 10^10 per orbit.
TEST_REL_SEP_TOL_PER_ORBIT = 1e-7 * 1e-3  # per orbit


def import_solver(solver):
    if solver == "ph4":
        from amuse.community.ph4.interface import ph4 as Gravity
    elif solver == "hermite":
        from amuse.community.hermite.interface import Hermite as Gravity
    elif solver == "huayno":
        from amuse.community.huayno.interface import Huayno as Gravity
    else:
        raise NotImplementedError("no import handling for this solver")
    return Gravity


@click.group()
@click.option("--solver", default="ph4", help="solver")
@click.option("--eps", default=0, help="softening length (pc)")
@click.option("--n_workers", default=1, help="number of workers")
@click.option("--timestep_parameter", default=0.01, help="solver timestep parameter")
@click.pass_obj
def cli(obj, solver, eps, n_workers, timestep_parameter):
    # Save basic parameters
    obj["solver"] = solver
    obj["eps"] = eps | units.parsec
    obj["n_workers"] = n_workers
    obj["timestep_parameter"] = timestep_parameter


@cli.command()
@click.option("--m_bh", default=1.0, help="BH mass (M_sun)")
@click.option("--m_imbh", default=1e3, help="IMBH mass (M_sun)")
@click.option("--d_0", default=1e-8, help="initial BH separation (pc)")
@click.option("--n_orbits", default=1000, help="number of orbits")
@click.option("--n_steps", default=100, help="number timesteps")
@click.pass_obj
def test(obj, m_bh, m_imbh, d_0, n_orbits, n_steps):
    # Set up bodies
    m_imbh = m_imbh | units.MSun
    m_bh = m_bh | units.MSun
    d_0 = d_0 | units.parsec
    v_0 = np.sqrt(units.constants.G * (m_bh + m_imbh) / d_0)
    m_tot = m_imbh + m_bh

    bodies = Particles(2)

    bodies[0].name = "IMBH"
    bodies[0].type = "black hole"
    bodies[0].mass = m_imbh
    bodies[0].position = [d_0 * m_bh / m_tot, 0.0 | units.parsec, 0.0 | units.parsec]
    bodies[0].velocity = [0.0 | units.kms, v_0 * m_bh / m_tot, 0.0 | units.kms]

    bodies[1].name = "BH"
    bodies[1].type = "black hole"
    bodies[1].mass = m_bh
    bodies[1].position = [-d_0 * m_imbh / m_tot, 0.0 | units.parsec, 0.0 | units.parsec]
    bodies[1].velocity = [0.0 | units.kms, -v_0 * m_imbh / m_tot, 0.0 | units.kms]

    end_time = 2 * np.pi * np.sqrt(d_0 ** 3 / units.constants.G / m_imbh) * n_orbits

    # Run the simulation
    main(
        obj["solver"],
        obj["eps"],
        obj["n_workers"],
        obj["timestep_parameter"],
        bodies,
        end_time,
        n_steps,
        TEST_SNAP_DIR,
    )

    # Load simulation data
    ts, z_imbhs, z_bhs, bh_separations, r_com_bhs, r_coms, Ks, Us = parse_files(
        TEST_SNAP_DIR, "hdf5", verbose=True
    )

    plot_diagnostics(
        ts,
        z_imbhs,
        z_bhs,
        bh_separations,
        r_com_bhs,
        r_coms,
        Ks,
        Us,
        join(TEST_FIG_DIR, "diagnostics.png"),
    )
    print("> Generated diagnostic plots")

    delta_sep_per_orbit = (
        np.max(np.abs((bh_separations | units.parsec) - d_0)) / d_0 / n_orbits
    )
    if delta_sep_per_orbit < TEST_REL_SEP_TOL_PER_ORBIT:
        print(
            "> TEST PASSED: BH separation maintained to "
            f"{delta_sep_per_orbit * 1000} per 1000 orbits, which is smaller "
            f"than required value of {TEST_REL_SEP_TOL_PER_ORBIT * 1000}"
        )
    else:
        raise ValueError(
            "TEST FAILED: BH separation maintained to "
            f"{delta_sep_per_orbit * 1000} per 1000 orbits, which is larger "
            f"than required value of {TEST_REL_SEP_TOL_PER_ORBIT * 1000}"
        )


@cli.command()
@click.option("--ic_file", default=None, help="initial condition file")
@click.option("--snap_dir", default="../snapshots")
@click.option("--end_time", default=100.0, help="end time (s)")
@click.option("--n_steps", default=100, help="number of timesteps")
@click.pass_obj
def run(obj, ic_file, snap_dir, end_time, n_steps):
    bodies = read_set_from_file(ic_file, "hdf5")
    print(f"> Loaded initial conditions for {len(bodies)} particles")

    end_time = end_time | units.s

    # Run the simulation
    main(
        obj["solver"],
        obj["eps"],
        obj["n_workers"],
        obj["timestep_parameter"],
        bodies,
        end_time,
        n_steps,
        snap_dir,
    )


def main(
    solver, eps, n_workers, timestep_parameter, bodies, end_time, n_steps, snap_dir
):
    delta_t = end_time / n_steps
    time = 0.0 | units.s
    bodies.time = time

    # Set scale of system based on BHs
    (imbh, bh), (i_imbh, i_bh) = get_bhs(bodies)
    bh_sep = (imbh.position - bh.position).length()
    m_tot = bodies.mass.sum()
    converter = nbody_system.nbody_to_si(m_tot, bh_sep)
    print("> Timestep (n-body units): ", converter.to_nbody(delta_t))

    Gravity = import_solver(solver)
    gravity = Gravity(converter, number_of_workers=n_workers, redirection="none")
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    print("> Created solver")

    # Set softening parameter
    gravity.parameters.epsilon_squared = eps ** 2
    print(f"> Softening length: {eps.number} pc")

    if solver in ["ph4", "huayno"]:
        gravity.parameters.timestep_parameter = timestep_parameter
    elif solver in ["hermite"]:
        gravity.parameters.dt_param = timestep_parameter
    print("> Set solver parameters")

    gravity.particles.add_particles(bodies)
    gravity.commit_particles()
    print("> Added particles")

    gravity.evolve_model(bodies.time)
    print("> Evolved model to time 0")

    # Channel to copy values from the code to the set in memory.
    channel = gravity.particles.new_channel_to(bodies)
    print("> Added channel")

    stopping_condition = gravity.stopping_conditions.collision_detection
    stopping_condition.enable()
    print("> Added stopping condition")

    # Run the simulation!
    print(f"Running simulation: {end_time.number} s in {n_steps} steps")
    for i in tqdm(range(n_steps)):
        time += delta_t
        gravity.evolve_model(time)

        # Make sure the following synchronizations don't lose a body
        n_bodies = len(bodies)

        # Synchronize to system with the solver to deal with adding/destroying
        # particles
        try:
            gravity.update_particle_set()
            gravity.particles.synchronize_to(bodies)
        except:
            pass

        # Copy particle parameter values from the module to the set in memory
        channel.copy()

        if stopping_condition is not None and stopping_condition.is_set():
            body_1 = stopping_condition.particles(0)[0]
            body_2 = stopping_condition.particles(1)[0]
            print("> stopping condition set at time {gravity.get_time().number} for:\n")
            print(body_1, body_2)
            raise NotImplementedError("collision handler has not been defined")

        # Make sure no bodies were lost
        if len(bodies) != n_bodies:
            raise ValueError("Number of bodies changed")

        # Output snapshot of bodies
        bodies.time = time
        write_set_to_file(bodies, f"{snap_dir}/snapshot_{i}.hdf5", "hdf5")

    gravity.stop()
    print("> Finished simulation")


if __name__ == "__main__":
    cli(obj={})
