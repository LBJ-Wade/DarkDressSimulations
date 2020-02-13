import click
from amuse.io import read_set_from_file, write_set_to_file
from amuse.units import nbody_system, units
from amuse.datamodel import Particles

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")

from matplotlib.animation import FuncAnimation


@click.command()
@click.option("--snap_dir",default="../snapshots")
@click.option("--n_snap", default=100, help="number of snapshots")
@click.option("--movie_file",default="../movies/movie.mp4", help="output filename of the movie")

def run(snap_dir, n_snap, movie_file):
    
    #Load in the first snapshot
    bodies = read_set_from_file(f"{snap_dir}/snapshot_0.hdf5", "hdf5")
    t = bodies.time[0].number
    
    
    
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    
    #Plot BH and DM
    p_BH = plt.scatter(bodies[0:2].x.number, bodies[0:2].y.number, color="b", marker='o',zorder=5)    
    p_DM = plt.scatter(bodies[2:].x.number, bodies[2:].y.number, s=2, color="k", marker='o')
    
    #Add timestamp
    timetext = plt.text(0.95, 0.95, f"t = {t} s", ha='right', va='top', transform=ax.transAxes)

    #Fix layout
    plt.xlim(-2e-8, 2e-8)
    plt.ylim(-2e-8, 2e-8)
    ax.set_aspect('equal')
    
    plt.xlabel(r'$x$ [pc]')
    plt.ylabel(r'$y$ [pc]')

    plt.tight_layout()

    #Animate
    anim = FuncAnimation(fig, animate, fargs=( timetext, p_BH, p_DM,snap_dir,),
                                   frames=n_snap)
    anim.save(movie_file, fps=40, bitrate=-1,codec='libx264',extra_args=['-pix_fmt', 'yuv420p'], dpi=300)


def animate(i, timetext, p_BH, p_DM, snap_dir):
    print(f" > snapshot_{i}")
    
    #Load in snapshot i
    bodies = read_set_from_file(f"{snap_dir}/snapshot_{i}.hdf5", "hdf5")
    
    #Update time
    t = bodies.time[0].number
    timetext.set_text(f"t = {t} s")
    
    #Get the (x,y) positions into some annoying format
    BH_data = np.vstack((bodies[0:2].x.number, bodies[0:2].y.number)).T
    DM_data = np.vstack((bodies[2:].x.number, bodies[2:].y.number)).T
    
    #Update scatter plot
    p_BH.set_offsets(BH_data)
    p_DM.set_offsets(DM_data)
    
    return timetext, p_BH, p_DM


if __name__ == "__main__":
    run()