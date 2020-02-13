from math import sqrt

import click
import matplotlib.pyplot as plt
import numpy as np
from amuse.datamodel import Particles
from amuse.io import read_set_from_file, write_set_to_file
#from amuse.community.ph4.interface import ph4 as Gravity
from amuse.lab import Huayno as Gravity
from amuse.units import nbody_system, units
from amuse.datamodel import Particles

from tqdm import tqdm


def print_log(time, gravity):
    # TODO: print out COM properties to make sure everything is ok
    # Compute total energy
    U = gravity.potential_energy
    T = gravity.kinetic_energy
    E = T + U
    print(f"[time = {time.number:.2f}] E_tot = {E.number}")


@click.command()
@click.option("--ic_file", default=None, help="initial condition file")
@click.option("--n_workers", default=1, help="number of workers")
@click.option("--softening_length", default=0, help="softening length (pc)")
@click.option("--accuracy_parameter", default=0.1)
@click.option("--manage_encounters", default=0)
@click.option("--end_time", default=100.)
@click.option("--n_steps", default=100)
@click.option("--snap_dir", default="../snapshots")
def run(
    ic_file,
    n_workers,
    softening_length,  # 1e-15 is a good value; 1e-10 looks questionable
    accuracy_parameter,
    manage_encounters,
    end_time,
    n_steps,
    snap_dir,
):
    print(units.constants.G.in_(units.parsec / units.MSun * units.kms ** 2))
    end_time = end_time | units.s
    delta_t = end_time / n_steps


    # Load initial conditions
    bodies = read_set_from_file(ic_file, "hdf5")
    print(f"> Loaded initial conditions for {len(bodies)} particles")
    
    # FIX ME
    #bodies = Particles(2)
    #bodies.add_particles(bodies_all[0:2])
    #bodies[1] = bodies_all[1]
    
    sep = (bodies[0].position - bodies[1].position).length()
    totmas = bodies.mass.sum()
    print("> Loaded initial conditions")

    converter = nbody_system.nbody_to_si(totmas, sep)

    gravity = Gravity(converter, number_of_workers=n_workers, redirection="none")
    gravity.initialize_code()
    gravity.parameters.set_defaults()
    print("> Created solver")

    # Set softening parameter
    softening_length = softening_length | units.parsec
    gravity.parameters.epsilon_squared = softening_length ** 2
    print(f"> Softening length: {softening_length.number} pc")

    #gravity.parameters.timestep_parameter = accuracy_parameter
    gravity.parameters.dt_param = accuracy_parameter
    gravity.parameters.manage_encounters = manage_encounters
    print("> Set solver parameters")

    gravity.particles.add_particles(bodies[:2])
    gravity.commit_particles()
    print("> Added particles")

    time = 0.0 | units.s
    gravity.evolve_model(time)
    bodies.time = time
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

        # Synchronize to system with the solver
        # TODO: why?
        try:
            gravity.update_particle_set()
            gravity.particles.synchronize_to(bodies)
        except:
            pass

        # Copy values from the module to the set in memory.
        channel.copy()

        if stopping_condition.is_set():
            body_1 = stopping_condition.particles(0)[0]
            body_2 = stopping_condition.particles(1)[0]
            print("> stopping condition set at time {gravity.get_time().number} for:\n")
            print(body_1, body_2)

        # Make sure no bodies were lost
        #if len(bodies) != n_bodies:
        #    raise ValueError("Number of bodies changed")

        # Output snapshot of bodies
        bodies.time = time
        write_set_to_file(bodies, f"{snap_dir}/snapshot_{i}.hdf5", "hdf5")

    gravity.stop()
    print("> Finished simulation")


if __name__ == "__main__":
    run()
