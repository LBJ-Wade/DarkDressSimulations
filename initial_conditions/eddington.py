#Eddington distribution function for the truncated density profile

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import hyp2f1
import matplotlib.pylab as pl
from scipy.integrate import quad
import units


import os.path


func = None
Menc_fun = None
psi_fun = None
r_sp = 1.0
M_BH = 0.0
gamma = 0.0
rho_sp = 1.0

alpha = 0.0
r_t = 0.0

rmin = 0.0
psi_interp = None


def loadDistribution(M_BH_in, rho0_in, gamma_in):
    global func, r_sp, rho_sp, M_BH, gamma, alpha, r_t, rmin, psi_interp
    
    M_BH = 1.0*M_BH_in
    rho_sp = 1.0*rho0_in
    gamma = 1.0*gamma_in
    
    #Characteristic size of the spike
    r_sp = ((3-gamma)*(0.2**(3.0-gamma))*M_BH/(2*np.pi*rho_sp))**(1.0/3.0)
    
    f_name = "distributions/distribution_M=" + str(int(M_BH)) + "_rho0=" + "{0:.2f}".format(rho_sp) + "_gamma=" + "{0:.2f}".format(gamma) + ".dat"
    f_data = np.loadtxt(f_name)

    #Read in the truncation radius and truncation slope from start of the file
    alpha = f_data[0,0]
    r_t = f_data[0,1]

    func = interp1d(f_data[1:,0], f_data[1:,1], kind='linear', bounds_error=False, fill_value=0.0)
    
    rmin = 6*M_BH*units.G_N/units.C_LIGHT**2
    

    f_potential = "distributions/potential_M=" + str(int(M_BH)) + "_rho0=" + "{0:.2f}".format(rho_sp) + "_gamma=" + "{0:.2f}".format(gamma) + ".dat"
    if (os.path.isfile(f_potential)):
        rlist, psilist = np.loadtxt(f_potential, unpack=True)
    else:
        print("    Tabulating potential from scratch...")

        rlist = np.logspace(np.log10(rmin), np.log10(50*r_sp),1000)
        psilist = np.asarray([psi_1(r) for r in rlist])
        
        htxt = "M_BH = " + str(int(M_BH)) + " M_solar, rho0 = " + "{0:.2f}".format(rho_sp) + "; gamma = " + "{0:.2f}".format(gamma)
        htxt += "\nalpha = " + str(alpha) + "; r_t = " + str(r_t) + " pc"
        htxt += "\nColumns: radius r [pc], psi(r) [(km/s)^2]"
        np.savetxt(f_potential, list(zip(rlist, psilist)), header=htxt)

    psi_interp = interp1d(rlist, psilist, bounds_error=False, fill_value=0.0)
    
    
# Density profile
def rho_DM(r):
    return rho_sp*(r/r_sp)**-gamma/(1+r/r_t)**alpha
   
# "True" density profile without truncation
def rho_DM_true(r):
    return rho_sp*(r/r_sp)**-gamma
    
#Enclosed mass
def Menc_fun(r):
    return (4*np.pi*rho_sp/(3-gamma))*r**3*(r/r_sp)**-gamma*hyp2f1(alpha, 3-gamma, 4-gamma, -r/r_t)

def Menc(r):
    return M_BH + np.clip(Menc_fun(r) - Menc_fun(rmin), 0, 1e30)
    
#Potential
@np.vectorize
def psi(r):
    #print(r)
    integ = lambda x: Menc(x)/x**2    
    if (r < r_sp):
        points = np.logspace(np.log10(10*r_sp), np.log10(r), 20)
    else: 
        points = None
    
    return -units.G_N*quad(integ, 100*r_sp, r, points=points)[0]


#Maximum speed at a given radius r
def vmax(r):
    return np.sqrt(2.0*psi_interp(r))

    
#Speed distribution f(v) at a given radius r
@np.vectorize
def f(r, v):
    if (v >= vmax(r)):
        return 0.0
    else:
        return 4.0*np.pi*(v**2)*func(psi_interp(r) - 0.5*v**2)/rho_DM(r)


