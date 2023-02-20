'''
Set up HPVsim and run with different parameters
'''


#%% General settings

# Standard imports
import hpvsim as hpv    # For running HPVsim
import numpy as np      # For numerics
import pandas as pd     # For data wrangling
import sciris as sc     # For utilities

# Example 1 (hello world): running the model with defaults
sim = hpv.Sim().run()
sim.plot()

# Modifying the hello world example
# The default parameter values are all recorded in hpvsim/parameters.py
# These can be modified by passing a parameter dictionary, or directly
# Example 2 (modifying pars directly):
sim_latent = hpv.Sim(hpv_control_prob=0.2) # 20% chance that infections are controlled latently

# Example 3 (using a pars dictionary):
pars = dict(
    location = 'kenya',
)
sim_kenya = hpv.Sim(pars=pars)

# It's time-consuming to run each of these individually, so instead let's run them in parallel
msim = hpv.MultiSim(sims=[sim_latent, sim_kenya])
msim.run()
msim.sims[0].plot()
msim.sims[1].plot()

# Exercise:
# Consider the outputs you got from the last plot. Now consider the parameters in parameters.py.
# What parameters do you think you might vary if you wanted cancers to peak at an older age?


