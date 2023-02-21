# Repo for scripts and data for HPVsim-GHL training session, February 2023

## Organization

The repository is organized as follows:

### Running scripts

There are three separate scripts for running and plotting simulations, calibrations, and scenarios. Specifically:

#### `run_sims.py`
 - This script can be used to run or plot a single simulation.

#### `run_calibration.py` 
This script is used for calibrating.
 - `run_calibration` section runs a small demonstration calibration, taking less than a minute to run
 - `run_calibrated_sim` runs a simulation using pre-generated parameters found from a more computationally intensive calibration run
 - `plot_calibration` creates plots of the calibration outputs.

#### `run_analyses.py`
 - `run_scenarios` compares 4 different scenarios, each with different screening algorithms in place from 2020-2060: HPV as a primary screen, AVE as a primary screen, HPV followed by VIA triage, and HPV followed by AVE triage.
 - `plot_scenarios` produces plots of the scenarios described above.


### Input data

- The `data` folder contains a series of .csv files tagged by country name, which contain the data that we use for calibrating the models. 


## Installation

If HPVsim is already installed (`pip install hpvsim`), no other dependencies are required.


## Usage

Run the desired analyses by running one of the scripts described above.


## Further information

Further information on HPVsim is available [here](http://docs.hpvsim.org). If you have further questions or would like technical assistance, please reach out to us at info@hpvsim.org.
