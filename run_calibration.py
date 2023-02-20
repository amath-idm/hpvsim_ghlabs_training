'''
Practice running calibrations with HPVsim
'''


#%% General settings

# Standard imports
import hpvsim as hpv    # For running HPVsim
import numpy as np      # For numerics
import pandas as pd     # For data wrangling
import sciris as sc     # For utilities
import pylab as pl      # For plotting
import seaborn as sns   # For plotting


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
        beta=[0.2, 0.1, 0.3],
        dur_transformed=dict(par1=[5, 3, 10]),
    )

    # Set up parameters to vary over genotypes
    dur_episomal = dict(par1=[4, 3, 10])  # Mean duration of dysplasia (years)
    sev_rate = [0.3, 0.2, 0.7]  # Logistic growth curve parameter (assumption)
    sev_infl = [13, 8, 20]
    transform_prob = [0.0001, 0.0001, 0.001]

    genotypes = ['hpv16', 'hpv18', 'hrhpv']
    genotype_pars = dict()
    for gtype in genotypes:
        genotype_pars[gtype] = dict()
        genotype_pars[gtype]['sev_rate'] = sev_rate
        genotype_pars[gtype]['dur_episomal'] = dur_episomal
        genotype_pars[gtype]['transform_prob'] = transform_prob
        genotype_pars[gtype]['sev_infl'] = sev_infl

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


def run_premade_calibration():
    '''
    Run a premade calibration for Nigeria
    '''

    # Parameters
    pars = dict(
        n_agents=50e3,
        dt=0.25,
        start=1950,
        end=2020,
        location='nigeria',
        debut= dict(
            f=dict(dist='normal', par1=14.8, par2=2.),
            m=dict(dist='normal', par1=17.0, par2=2.)),
        genotypes=[16, 18, 'hrhpv'],
        condoms=dict(m=0.01, c=0.1, o=0.2),
        eff_condoms=0.5,
        ms_agent_ratio=100,
    )

    # Make sim
    sim = hpv.Sim(pars=pars)

    # Load pre-made calibration parameters
    file = 'results/nigeria_pars_prerun.obj'
    calib = sc.loadobj(file)

    # Initialize the sim, then update the parameters
    sim.initialize()
    sim.update_pars(calib)

    # Run and plot
    sim.run()
    sim.plot()

    # Plot calibration
    calib.plot()

    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    run_calibration = False # Takes ~30seconds
    load_calibration = True # Takes ~3min

    if run_calibration: sim, calib = run_calibration()
    if load_calibration: sim, calib = run_premade_calibration()

    sc.toc(T)
    print('Done.')

