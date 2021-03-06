## DarkDressSimulations - Initial conditions

Code for generating initial conditions. Run `python GenerateICs_binary.py` to generate initial conditions for a binary with a DM halo around the central BH. Run `python GenerateICs_single.py` to just get ICs for the central BH and DM spike (with no orbiting compact object). Run either of the scripts with the flag `--help` to check the commandline options.

The initial DM density profile is given in Eq. (B1) of the draft.

**BJK:** For the moment, I've only uploaded the distribution function for the case of a central BH of mass 1000 Msun. I have some separate code for tabulating the distribution functions, which I will upload soon. But for now, this should be enough to get started. I also haven't looked into how AMUSE handles initial conditions, so the code just outputs a list of masses and phase space coordinates (to the file `IC.txt`).

- `distributions` contains tabulated distribution functions f(E) and corresponding potentials psi(r) for the DM halos we're using.
- `halos` contains files with the phase space coordinates of 100,000 randomly sampled particles from the underlying distribution functions. Set the `haloID` parameter internally in the `python GenerateICs_XXX.py` files to draw a subset of the particles from these files (for speed). Or if you set `haloID = None`, the samples can be generated from scratch.

