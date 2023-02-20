'''
Practice running calibrations with HPVsim
'''


#%% General settings

# Standard imports
import hpvsim as hpv    # For running HPVsim
import numpy as np      # For numerics
import pandas as pd     # For data wrangling
import sciris as sc     # For utilities


#%% Set up a calibration example
def run_calibration():

    sc.heading('Testing calibration')

    import hpvsim as hpv

    # Configure a simulation with some parameters
    pars = dict(n_agents=10e3, start=1980, end=2020, dt=0.25, location='nigeria')
    sim = hpv.Sim(pars)

    # Specify some parameters to adjust during calibration.
    # The parameters in the calib_pars dictionary don't vary by genotype,
    # whereas those in the genotype_pars dictionary do. Both kinds are
    # given in the order [best, lower_bound, upper_bound].
    calib_pars = dict(
        beta=[0.05, 0.010, 0.20],
        dur_transformed=dict(par1=[5, 3, 10]),
    )

    genotype_pars = dict(
        hpv16=dict(
            sev_rate=[0.5, 0.2, 1.0],
            dur_episomal=dict(par1=[6, 4, 12])
        ),
        hpv18=dict(
            sev_rate=[0.5, 0.2, 1.0],
            dur_episomal=dict(par1=[6, 4, 12])
        )
    )

    # List the datafiles that contain data that we wish to compare the model to:
    datafiles = ['data/nigeria_cancer_cases.csv',
                 'data/nigeria_cancer_types.csv']

    # List extra results that we don't have data on, but wish to include in the
    # calibration object so we can plot them.
    results_to_plot = ['cancer_incidence', 'asr_cancer_incidence']

    # Create the calibration object, run it, and plot the results
    calib = hpv.Calibration(
        sim,
        calib_pars=calib_pars,
        genotype_pars=genotype_pars,
        extra_sim_results=results_to_plot,
        datafiles=datafiles,
        total_trials=4, n_workers=4
    )
    calib.calibrate(die=True)
    calib.plot(res_to_plot=4)

    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, calib = run_calibration()

    sc.toc(T)
    print('Done.')

