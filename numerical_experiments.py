# Script used to regenerate Figure 1 of the paper "Joint Sparsity with Partially Known Support and Application to
# Ultrasound Imaging" submitted to IEEE Signal Processing Letters
# Author: adrien.besson@epfl.ch
# Date: July 2018

import pickle
import os.path
import utils as ut

# Figure one wants to regenerate
fig_name = 'f' # 'a' to 'e'
flag_plot = 1 # 1 for plotting, 0 for no plotting
flag_save = 1 # 1 for saving, 0 for no saving
flag_run_experiments = 0 # 1=run the experiments, 0=use stored files

# Number of simulation runs
num_trial = 1000

# Dimension of the problem (X is a nxN matrix)
n = 128
N = 30

# Row-sparsity of X
k = 30

# Algorithms to be tested
list_algorithms = ['SA-MUSIC', 'RA-ORMP', 'SNIHT']

# Set the parameters depending on the desired figure
if fig_name == 'a' or fig_name == 'b':
    snr = 1000
    m = range(30, 90, 4)
    s = [10]
    filename_results = 'experiment_fig_' + 'a' + '.pickle'
elif fig_name == 'c':
    snr = 1000
    m = [51]
    s = range(1, 30, 2)
    filename_results = 'experiment_fig_' + 'c' + '.pickle'
elif fig_name == 'd' or fig_name == 'e':
    snr = 30
    m = range(30, 90, 4)
    s = [10]
    filename_results = 'experiment_fig_' + 'd' + '.pickle'
elif fig_name == 'f':
    snr = 30
    m = [51]
    s = range(1, 30, 2)
    filename_results = 'experiment_fig_' + 'f' + '.pickle'

# Run the experiments
dir_results = os.path.join(os.getcwd(), 'results')
if flag_run_experiments:
    # Run the experiments
    proba_recovery, algorithms= ut.run_numerical_experiment(num_trial, n, N, m, k, s, snr, list_algorithms)
    # Create the dictionary containing the results
    result_dict = {
        'proba_recovery' : proba_recovery[0],
        'proba_recovery25': proba_recovery[1],
        'proba_recovery50': proba_recovery[2],
        'proba_recovery75': proba_recovery[3],
        'proba_recovery90': proba_recovery[4],
        'measurements': m,
        'sparsity': k,
        'n': n,
        'N': N,
        'rank':s,
        'snr':snr,
        'num-trial': num_trial,
        'algorithm': algorithms
    }
    # Check if results folder exists
    if not(os.path.exists(dir_results)):
        os.mkdir(dir_results)

    # Save
    pickle.dump(result_dict, open(os.path.join(dir_results, filename_results), "wb"))
else:
    result_dict = pickle.load(open(os.path.join(dir_results, filename_results), "rb" ))

# Plot the figure
proba_recovery_pks = {}
proba_recovery_pks['25'] = result_dict['proba_recovery25']
proba_recovery_pks['50'] = result_dict['proba_recovery50']
proba_recovery_pks['75'] = result_dict['proba_recovery75']
proba_recovery_pks['90'] = result_dict['proba_recovery90']
if fig_name == 'a' or fig_name == 'd':
    ut.plot_fig_spl(result_dict['proba_recovery'], proba_recovery_pks, result_dict['measurements'], result_dict['rank'], fig_name, result_dict['algorithm'], flag_plot, flag_save)

else:
    ut.plot_fig_spl(result_dict['proba_recovery'], proba_recovery_pks['75'], result_dict['measurements'], result_dict['rank'], fig_name, result_dict['algorithm'], flag_plot, flag_save)
