

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


#Parse the arguments!
parser = argparse.ArgumentParser(description='...')
parser.add_argument('-M_BH', '--M_BH', help='Mass of central BH in M_sun', type=float, default = 1e3)
#parser.add_argument('-M_NS', '--M_NS', help='Mass of orbiting compact object in M_sun', type=float, default = 1.0)
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

M2 = 0.0


print("  System properties (lonely BH):")

print("    M_BH1 [M_sun]:", M_BH)
print("    r_isco [pc]:", 6*M_BH*units.G_N/units.C_LIGHT**2)
print("    Total Halo Mass [M_sun]:", edd.Menc(edd.r_sp) - M_BH)
print("    Mass inside BH separation (M_sun):",edd.Menc(d0))

print(" ")

print("  Generating ICs with %d DM pseudo-particles..."%nDM)


#Set the haloID in order to load the particles from file rather than sampling them from scratch
#haloID = "halo_M=" + str(int(M_BH)) + "_rho0=" + "{0:.2f}".format(rho0) + "_gamma=" + "{0:.2f}".format(gamma)
haloID = None

#Coordinates of central BH
x1 = [d0*M2/(M1+M2), 0, 0]
v1 = [0, 0, 0]


mlist, xlist, vlist = BH.AddDressedBH(x1, v1,  M_BH, rho0, gamma, nDM, haloID=haloID, verbose=True)

if (NO_DM):
    mlist[1:] = 0.0*mlist[1:] + 1e-15


#Order by mass
inds = (mlist.argsort())[::-1]
xlist = xlist[inds,:]
vlist = vlist[inds,:]
mlist = mlist[inds]

#Save to file
headertxt = "Columns: M [M_sun], x [pc], y [pc], z [pc], vx [km/s], vy [km/s], vz [km/s]"
np.savetxt("IC.txt", list(zip(mlist, xlist[:,0],xlist[:,1],xlist[:,2],vlist[:,0],vlist[:,1],vlist[:,2])), header=headertxt)


#The rest of this file uses pyGadgetIC to output
#an IC file in the format required by Gadget
#see https://github.com/ldocao/pygadgetic
"""

cnt = Counter(mlist_sorted)

mvals = np.array([k for k, v in cnt.items()])
n = np.array([v for k, v in cnt.items()])

n_species = len(mvals)
n_particles = np.zeros(n_species, dtype='int')
n_particles[0] = 1
n_particles[1] = nDM
n_particles[2] = 1
#n_particles = n[mvals.argsort()[::-1]]
if (np.sum(n) != nDM + 2):
    raise ValueError("Number of particles doesn't seem to match...")

##define number of particles
npart = np.zeros(6, dtype='int')
npart[1:(n_species+1)] = n_particles

print("  Number of particles:", npart)

#for i in range(N_shell):
#    npart[i+2] = nDM_shell[i]

total_number_of_particles=np.sum(npart) #total number of particles

##create objects
my_header=pygadgetic.Header()
my_body=pygadgetic.Body(npart)

my_body.pos = xlist_sorted
my_body.mass = mlist_sorted
my_body.vel = vlist_sorted

#------------
#Setting the DM mass to something small
#my_body.mass[my_body.mass < 1e-3] =1e-12 +  0.0*my_body.mass[my_body.mass < 1e-3]

print("  Particles masses:", my_body.mass)

#Checking CoM properties
#print("   CoM position [pc]:", np.sum(np.atleast_2d(my_body.mass).T*my_body.pos*1e-5, axis=0)/np.sum(my_body.mass))
#print("   CoM velocity [pc/kyr]:", np.sum(np.atleast_2d(my_body.mass).T*my_body.vel, axis=0)*3.24078e-14*(3600*24.0*365*1000)/np.sum(my_body.mass))

print("   ")
#print("   v_rms [km/s]:", np.mean(np.sqrt(np.sum(my_body.vel**2, axis=1))))

#PBH.AddDressedPBH(my_body,np.arange(0,nDM),-1, nDM, [0, 0, 0],[0, 0, 0], r_soft, a, haloID=haloID1, verbose=True)
#print "  Second Halo:"
#PBH.AddDressedPBH(my_body,np.arange(nDM,2*nDM),-1, nDM, [-apo/2.0, 0, 0],[0, -vapo, 0], x_soft,a,  haloID=haloID2,verbose=True)

##fill in the header
my_header.NumPart_ThisFile = np.array(npart)
my_header.NumPart_Total = np.array(npart)

#id
my_body.id[:]=np.arange(0,total_number_of_particles) #generate an array from 0 to total_number_of_particles

print("  Printing to file...")
##now writes the initial condition file
#try:
my_name="../run/EMRI1.dat"
pygadgetic.dump_ic(my_header,my_body,my_name)
#except IOError:
#    my_name="run/PBH1.dat"
#    pygadgetic.dump_ic(my_header,my_body,my_name)

"""


