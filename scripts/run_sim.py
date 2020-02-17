import warnings
from math import sqrt

import click
import matplotlib.pyplot as plt
import numpy as np
from amuse.datamodel import Particles
from amuse.io import read_set_from_file, write_set_to_file
from amuse.support.exceptions import AmuseWarning
from amuse.units import nbody_system, units
from tqdm import tqdm

from utils import get_bhs

# Ignore citation warnings
warnings.filterwarnings("ignore", category=AmuseWarning)


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


@click.command()
@click.option("--ic_file", default=None, help="initial condition file")
@click.option("--snap_dir", default="../snapshots")
@click.option("--bhs_only", default=False, help="run tests with only the black holes")
@click.option("--solver", default="ph4", help="solver")
@click.option("--eps", default=0, help="softening length (pc)")
@click.option("--n_workers", default=1, help="number of workers")
@click.option("--timestep_parameter", default=0.03, help="solver timestep parameter")
@click.option("--end_time", default=100.0, help="end time (s)")
@click.option("--n_steps", default=100, help="number of timesteps")
def run(
    ic_file,
    snap_dir,
    bhs_only,
    solver,
    eps,
    n_workers,
    timestep_parameter,
    end_time,
    n_steps,
):
    # Dynamically import the solver
    Gravity = import_solver(solver)

    # Give parameters units
    time = 0.0 | units.s
    eps = eps | units.parsec
    end_time = end_time | units.s
    delta_t = end_time / n_steps

    # Load initial conditions
    bodies = read_set_from_file(ic_file, "hdf5")
    (imbh, bh), (i_imbh, i_bh) = get_bhs(bodies)
    if bhs_only:
        bodies = bodies[[i_imbh, i_bh]]
        print("> Loaded initial conditions for black holes only")
    else:
        print(f"> Loaded initial conditions for {len(bodies)} particles")
    bodies.time = time

    print("> Loaded initial conditions")

    # Set scale of system based on BHs
    bh_sep = (imbh.position - bh.position).length()
    m_tot = bodies.mass.sum()
    converter = nbody_system.nbody_to_si(m_tot, bh_sep)
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

    gravity.evolve_model(time)
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
    run()
