import numpy as np
#import pygadgetic

import units
import math

from tqdm import tqdm

import os.path
import os
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d

from scipy.optimize import brenth

from matplotlib import pylab as pl
import sys



#-------------
#You should just be able to import whichever eddington module
#you're interested in and everything should work from here on...
import eddington as edd

#-------------


def AddDressedBH(x0, v0, M_BH, rho0, gamma, nDM = 100, verbose=False, haloID=None):
    """Add a dressed PBH to the initial conditions...
    
    Parameters:
        x0     - initial position of BH (in pc)
        v0     - initial velocity of BH+DM halo (in km/s)
        M_BH   - BH mass in M_sun
    
        rho0   - characteristic spike density in M_sun/pc^3
        gamma  - spike power-law slope
    
        nDM    - number of DM particles around this BH

        haloID - string identifying a text file to load the halo from (in folder /halos)
                 If file not found, a new halo is generated and saved in /halos.
                 Set haloID = None (default) to ignore this option.
    """
 
    #For now, don't edit this:
    fix_CoM = False

    #BH mass and truncation radius imported from the eddington file for
    #self-consistency
    edd.loadDistribution(M_BH, rho0, gamma)
    

    r_isco = 6*M_BH*units.G_N/units.C_LIGHT**2
    
    #Minimum and maximum radius to initialise particles down to
    r_min = r_isco
    r_max = 50*edd.r_sp 
    
    
    mHalo = edd.Menc(r_max) - edd.Menc(r_min)

    #Initialise the masses, positions and velocities
    m_vals = np.zeros(nDM + 1)
    pos_vals = np.zeros((nDM + 1,3))
    vel_vals = np.zeros((nDM + 1,3))
    
    #Set up the central black hole
    m_vals[0] = M_BH
    pos_vals[0,:] = np.zeros(3)
    vel_vals[0,:] = np.zeros(3)
    
    
    #PBH position and velocity (before CoM velocity is subtracted...)
    xPBH=np.array([0.,0.,0.])
    vPBH=np.array([0.,0.,0.])
    
    if (haloID is not None):
        halofile = "halos/" + haloID + ".txt"
        
    #Check to see whether a halo file already exists...
    if ((haloID is not None) and os.path.isfile("halos/" + haloID + ".txt")):
        print("    Loading halo from file. HaloID:", haloID)
        
        #Load DM phase space coordinates from file
        halo_data = np.loadtxt(halofile)

        nStored = len(halo_data)
        #print(nStored)
        assert nStored >= nDM
        
        inds = np.random.choice(nStored, size=nDM, replace=False)
        xvals = halo_data[inds,0]
        yvals = halo_data[inds,1]
        zvals = halo_data[inds,2]

        pos_vals[1:,:] = np.array([xvals, yvals, zvals]).T
        
        vxvals = halo_data[inds,3]
        vyvals = halo_data[inds,4]
        vzvals = halo_data[inds,5]
        
        vel_vals[1:,:] = np.array([vxvals, vyvals, vzvals]).T

        m_vals[1:] = mHalo/nDM

    else:
        if (haloID is not None):
            print(("   Halo file <" + halofile+"> not found. Generating from scratch..."))
    
        #Generate the mass profile
        print("    Generating mass profile...")
    
        rlist = np.logspace(np.log10(r_min), np.log10(r_max), 1000)
        #rlist = np.logspace(np.log10(r_isco), np.log10(r_max), 1000)
        Menc = 0.0*rlist

        Min = edd.Menc(r_min)
        for i in range(len(rlist)):
            Menc[i] = edd.Menc(rlist[i]) - Min
            
        Menc -= Menc[0]
        M_max = Menc[-1]

        M_interp = interp1d(Menc/M_max, rlist, kind='linear')
        
        #DM positions        
        rvals = M_interp(np.random.rand(nDM))
        
        #Generate some random directions for setting particle positions
        ctvals = 2.0*np.random.rand(nDM) - 1.0
        thetavals = np.arccos(ctvals)
        phivals = 2*np.pi*np.random.rand(nDM)

        xvals = rvals*np.cos(phivals)*np.sin(thetavals)
        yvals = rvals*np.sin(phivals)*np.sin(thetavals)
        zvals = rvals*np.cos(thetavals)

        pos_vals[1:,:] = np.array([xvals, yvals, zvals]).T
        m_vals[1:] = mHalo/nDM
 
        
        #Generate velocities
        vvals = np.zeros(nDM)
        for ind in tqdm(range(nDM), desc="    Sampling velocities..."):
            count = 0
            r = rvals[ind]
            #Now sample f(v) at given r to get the speed v
            found = 0
    
            while (found == 0):
                count += 1
                if (count > 100000):
                    print("Velocity sampling failed at r = ", r)
                    sys.exit()

                #Rejection sampling for the velocities
                v = np.random.rand(1)*edd.vmax(r)
                #Use 5.0/vmax as the 'maximum' values of f(v)
                #but in some cases it might not be enough...
                ratio = 5.0
                if (np.random.rand(1)*(ratio/edd.vmax(r)) < edd.f(r, v)):
                    found = 1
                    vvals[ind] = v

        #Get a new set of random directions for the velocities
        ctvals2 = 2.0*np.random.rand(nDM) - 1.0
        thetavals2 = np.arccos(ctvals2)
        phivals2 = 2*np.pi*np.random.rand(nDM)

        vxvals = vvals*np.cos(phivals2)*np.sin(thetavals2)
        vyvals = vvals*np.sin(phivals2)*np.sin(thetavals2)
        vzvals = vvals*np.cos(thetavals2)


        vel_vals[1:,:] = np.array([vxvals, vyvals, vzvals]).T

    
        #Save the output to a halo file if needed
        if (haloID is not None):
            headertxt = "Number of DM particles: " + str(nDM)
            headertxt += "\nColumns: x [pc], y [pc], z [pc], vx [km/s], vy [km/s], vz [km/s]"
            np.savetxt("halos/" + haloID + ".txt", list(zip(pos_vals[1:,0],pos_vals[1:,1],pos_vals[1:,2],vel_vals[1:,0],vel_vals[1:,1],vel_vals[1:,2])), header=headertxt)
    
        
    #Subtract off net position and momentum of system
    if (fix_CoM):
        #Deal with C.o.M.    
        totmass = np.sum(m_vals)
        #print("    Total Mass [Halo + BH]:", totmass)
        x_CoM = np.zeros(3)
        x_CoM[0] = np.sum(pos_vals[:,0]*m_vals)
        x_CoM[1] = np.sum(pos_vals[:,1]*m_vals)
        x_CoM[2] = np.sum(pos_vals[:,2]*m_vals)
        x_CoM /= totmass
        
        p_CoM = np.zeros(3)
        p_CoM[0] = np.sum(vel_vals[:,0]*m_vals)
        p_CoM[1] = np.sum(vel_vals[:,1]*m_vals)
        p_CoM[2] = np.sum(vel_vals[:,2]*m_vals)

        vel_vals -= p_CoM/totmass
        pos_vals -= x_CoM

        #Now rotate so that IMBH is in x-y plane
        #x_BH = pos_vals[0,:]
        #print(x_BH)
        #theta_rot = np.arctan(-x_BH[2]/x_BH[1])
        #print(theta_rot)
        #pos_old = 1.0*pos_vals
        #pos_vals[:,0] = pos_old[:,0] #Keep x coordinates the same
        #pos_vals[:,1] = np.cos(theta_rot)*pos_old[:,1] - np.sin(theta_rot)*pos_old[:,2]
        #pos_vals[:,2] = np.sin(theta_rot)*pos_old[:,1] + np.cos(theta_rot)*pos_old[:,2]
    
    
    #Add on the CoM position and velocity
    pos_vals += np.asarray(x0)
    vel_vals += v0
    
    return m_vals, pos_vals, vel_vals
