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

# Modifying the hellow world example
# The default parameter values are all recorded in hpvsim/parameters.py
# These can be modified by passing a parameter dictionary, or directly
# Example 2 (modifying pars directly):
sim_latent = hpv.Sim(hpv_control_prob=0.2) # 20% chance that infections are controlled latently
sim_latent.run()
sim_latent.plot()

# Example 3 (using a pars dictionary):
pars = dict(
    location = 'kenya',
    n_agents = 5e3,
)
sim_kenya = hpv.Sim(pars=pars)
sim_kenya.run()
sim_kenya.plot()