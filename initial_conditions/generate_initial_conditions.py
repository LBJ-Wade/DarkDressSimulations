

import numpy as np
#import pygadgetic
import units
#-------------

import eddington as edd
from scipy.interpolate import interp1d
import argparse


from collections import Counter

#This is the module that add the BHs to the initial condition file
import BH
import amuse.units.units as amuse_units
from amuse.lab import Particles, write_set_to_file


def save_data_to_file(mlist, xlist, vlist):
    bodies = Particles(len(inds))
    bodies.name = "dmp"
    bodies.type = "wimp"
    bodies.mass = mlist | amuse_units.MSun
    bodies[0].name = "SMBH"
    bodies[0].type = "black hole"
    bodies[1].name = "IMBH"
    bodies[1].type = "black hole"
    bodies.position = xlist | amuse_units.parsec
    bodies.velocity = vlist  | amuse_units.kms
    write_set_to_file(bodies, "initial_conditions.amuse", "hdf5")

#Parse the arguments!
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-M_BH', '--M_BH', help='Mass of central BH in M_sun', type=float, default = 1e3)
parser.add_argument('-M_NS', '--M_NS', help='Mass of orbiting compact object in M_sun', type=float, default = 1.0)
parser.add_argument('-N_DM','--N_DM', help='Number of DM particles', type=int, default=1000)
parser.add_argument('-d0','--d0', help='Initial separation in pc', type=float, default=1.0e-8)
parser.add_argument('-NO_DM', '--NO_DM', help='Set flag 1 to fix the DM particle mass to zero', type=int, default=0)

args = parser.parse_args()

nDM = args.N_DM
d0 = args.d0
M_BH = args.M_BH
NO_DM = args.NO_DM

# Some baseline parameters which we always seem to keep fixed:
rho0 = 226.0 #Characteristic DM density in M_sun/pc^3
gamma = 7.0/3.0 #Power-law slope

edd.loadDistribution(M_BH, rho0, gamma)

if (NO_DM):
    M1 = M_BH
else:
    M1 = edd.Menc(d0)

M2 = args.M_NS

print("  System properties (binary):")

print("    M_BH1 [M_sun]:", M_BH)
print("    M_BH2 [M_sun]:", M2)
print("    Initial separation [pc]:", d0)
print("    Orbital period [s]:", 2*np.pi*np.sqrt(d0**3*(units.L_pc)**2/(units.G_N*M_BH)))
print("    r_isco [pc]:", 6*M_BH*units.G_N/units.C_LIGHT**2)
print("    Total Halo Mass [M_sun]:", edd.Menc(edd.r_sp) - M_BH)
print("    Mass inside BH separation (M_sun):",edd.Menc(d0))

print(" ")

print("  Generating ICs with %d DM pseudo-particles..."%nDM)

#Set the haloID in order to load the particles from file rather than sampling them from scratch
#haloID = "halo_M=" + str(int(M_BH)) + "_rho0=" + "{0:.2f}".format(rho0) + "_gamma=" + "{0:.2f}".format(gamma)
haloID = None

#Orbital parameters (for circular orbit)

#Relative orbital speed
v0 = np.sqrt(units.G_N*(M1 + M2)/d0)


#Coordinates of central BH
x1 = [d0*M2/(M1+M2), 0, 0]
v1 = [0, v0*M2/(M1 + M2), 0]

#Coordinates of orbiting compact object
x2 = [-d0*M1/(M1+M2), 0, 0]
v2 = [0, -v0*M1/(M1 + M2), 0]

mlist, xlist, vlist = BH.AddDressedBH(x1, v1,  M_BH, rho0, gamma, nDM, haloID=haloID, verbose=True)

if (NO_DM):
    mlist[1:] = 0.0*mlist[1:] + 1e-15


#Add the other BH
mlist = np.append(mlist, args.M_NS)
xlist = np.append(xlist, np.atleast_2d(x2), axis=0)
vlist = np.append(vlist, np.atleast_2d(v2), axis=0)

#Order by mass
inds = (mlist.argsort())[::-1]
xlist = xlist[inds,:]
vlist = vlist[inds,:]
mlist = mlist[inds]

save_data_to_file(mlist, xlist, vlist)
