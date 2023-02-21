'''
Practice running analyses with HPVsim
'''


#%% General settings

# Standard imports
import hpvsim as hpv    # For running HPVsim
import numpy as np      # For numerics
import pandas as pd     # For data wrangling
import sciris as sc     # For utilities
import pylab as pl      # For plotting

# Comment out to not run
to_run = [
    # 'run_scenarios',
    'plot_scenarios',
]

#%% Functions
def get_screen_intvs(primary=None, triage=None, start_year=2020, end_year=2040, sim_end_year=2060):
    ''' Make interventions for screening scenarios '''

    # Create AVE products
    if primary=='ave':
        df = pd.read_csv('ave_pars.csv')
        primary = hpv.dx(df)
    elif triage == 'ave':
        df = pd.read_csv('ave_pars.csv')
        triage = hpv.dx(df)

    # Define gradual scale-up of screening
    screen_ramp = np.arange(start_year, end_year, dtype=int) # Ramp-up years
    screen_prob_final = 0.7
    treat_prob = 0.9
    triage_prob = 0.9
    screen_coverage = list(np.linspace(start=0, stop=screen_prob_final, num=len(screen_ramp)))
    screen_coverage += [screen_prob_final] * (sim_end_year - end_year + 1)

    # Routine screening
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 10 / sim['dt']))
    screening = hpv.routine_screening(
        product=primary,
        prob=screen_coverage,
        eligibility=screen_eligible,
        age_range=[30, 50],
        start_year=start_year,
        label='screening'
    )

    if triage is not None:

        # Triage screening
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        triage_screening = hpv.routine_triage(
            start_year=start_year,
            prob=triage_prob,
            annual_prob=False,
            product=triage,
            eligibility=screen_positive,
            label='triage'
        )
        triage_positive = lambda sim: sim.get_intervention('triage').outcomes['positive']
        assign_treatment = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product='tx_assigner',
            eligibility=triage_positive,
            label='tx assigner'
        )
    else:
        # Assign treatment
        screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
        assign_treatment = hpv.routine_triage(
            start_year=start_year,
            prob=1.0,
            annual_prob=False,
            product='tx_assigner',
            eligibility=screen_positive,
            label='tx assigner'
        )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob=treat_prob,
        annual_prob=False,
        product='ablation',
        eligibility=ablation_eligible,
        label='ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                             sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob=treat_prob,
        annual_prob=False,
        product='excision',
        eligibility=excision_eligible,
        label='excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob=treat_prob,
        annual_prob=False,
        product=hpv.radiation(),
        eligibility=radiation_eligible,
        label='radiation'
    )

    if triage is not None:
        st_intvs = [screening, triage_screening, assign_treatment, ablation, excision, radiation]
    else:
        st_intvs = [screening, assign_treatment, ablation, excision, radiation]

    return st_intvs


def run_calibrated_sim(interventions=None, label=None, meta=None):
    '''
    Run a premade calibration for Nigeria
    '''

    # Parameters
    pars = dict(
        n_agents=50e3,
        dt=0.25,
        start=1950,
        end=2060,
        location='nigeria',
        debut= dict(
            f=dict(dist='normal', par1=14.8, par2=2.),
            m=dict(dist='normal', par1=17.0, par2=2.)),
        genotypes=[16, 18, 'hrhpv'],
        condoms=dict(m=0.01, c=0.1, o=0.2),
        eff_condoms=0.5,
        ms_agent_ratio=100,
    )

    # Optionally add interventions
    if interventions is not None:
        sim = hpv.Sim(pars=pars, interventions=interventions, label=label)
    else:
        sim = hpv.Sim(pars=pars, label=label)

    # Load pre-made calibration parameters
    calib_pars_file = 'results/nigeria_pars_prerun.obj'
    calib_pars = sc.loadobj(calib_pars_file)

    # Initialize the sim, then update the parameters
    sim.initialize()
    sim.update_pars(calib_pars)
    sim.run()

    return sim


def run_scens(screen_scens=None):
    '''
    Run screening/triage product scenarios
    '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(screen_scens)

    for i_sc, scen_label, screen_scen_pars in screen_scens.enumitems():
        screen_intvs = get_screen_intvs(**screen_scen_pars)
        count += 1
        meta = sc.objdict()
        meta.count = count
        meta.n_sims = n_sims
        meta.inds = [i_sc]
        meta.vals = sc.objdict(sc.mergedicts(screen_scen_pars, dict(scen_label=scen_label)))
        ikw.append(sc.objdict(interventions=screen_intvs, label=scen_label))
        ikw[-1].meta = meta

    # Actually run
    sc.heading(f'Running {len(ikw)} scenario sims...')
    all_sims = sc.parallelize(run_calibrated_sim, iterkwargs=ikw)

    return all_sims


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    sc.options(fontsize=20)

    #################################################################
    # RUN AND PLOT SCENARIOS
    #################################################################
    # Run scenarios (takes 3-5 min)

    if 'run_scenarios' in to_run:

        filestem = 'screening_results'

        # Construct the scenarios
        # Baseline scenarios    : No screening, HPV, VIA, HPV+VIA
        # AVE as primary screen : AVE
        # AVE as triage         : HPV+AVE
        screen_scens = sc.objdict({
            # 'No screening': dict(),
            'HPV': dict(primary='hpv'),
            'AVE': dict(primary='ave'),
            'HPV+VIA': dict(primary='hpv', triage='via'),
            'HPV+AVE': dict(primary='hpv', triage='ave'),
        })

        all_sims = run_scens(screen_scens=screen_scens)

        # Save total cancers for plotting
        results = {}
        rn = 0
        for scen_label in screen_scens.keys():
            results[scen_label] = all_sims[rn].results
            rn += 1

        sc.saveobj(f'results/screening_results.obj', results)


    # Plot results of scenarios
    if 'plot_scenarios' in to_run:

        colors = sc.gridcolors(10) # Define a set of distinct colors to use for each scenarios
        scens = ['HPV', 'AVE', 'HPV+VIA', 'HPV+AVE']
        screening_results = sc.loadobj(f'results/screening_results.obj')
        result_name = 'asr_cancer_incidence'

        fig, ax = pl.subplots(figsize=(16, 10)) # Create a plot
        years = screening_results['HPV']['year'][50:] # Only plotting from 2000 onwards
        for sn, scen_label in enumerate(scens):
            ydata = screening_results[scen_label][result_name][50:]
            ax.plot(years, ydata, color=colors[sn], label=scen_label)
            ax.legend(bbox_to_anchor=(1.05, 0.8), fancybox=True)
            sc.SIticks(ax)
            ax.set_title(f'{screening_results[scen_label][result_name].name}, Nigeria')
            fig.tight_layout()
            fig_name = f'figs/{result_name}.png'
            sc.savefig(fig_name, dpi=100)



    print('Done.')

