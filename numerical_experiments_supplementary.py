# Script used to regenerate the results in the supplementary material of the paper "Joint Sparsity with Partially Known Support abd Application to
# Ultrasound Imaging" submitted to IEEE Signal Processing Letters
# Author: adrien.besson@epfl.ch
# Date; July 2018

import pickle
import os.path
import numpy as np
import subspacemethods.music as music
import matplotlib.pyplot as plt
import utils as ut

# Parameters related to the simulation
flag_plot = 1 # 1 for plotting, 0 for no plotting
flag_save = 0 # 1 for saving, 0 for no saving
flag_run_experiments = 0 # 1=run the experiments, 0=use stored files
filename_results = 'experiment_supplementary'

# Directory containing the results
dir_results = os.path.join(os.getcwd(), 'results')

if flag_run_experiments:
    # Number of simulation runs
    num_trial = 1000

    # Dimension of the problem (X is a nxN matrix)
    n = 64
    N = 128
    k = 16
    n_pks = 8
    n_experiments = 2
    list_m = np.linspace(16, 48).astype(int)
    nm = len(list_m)

    # Output variables
    proba_recovery_music_pks = np.zeros(shape=(n_experiments, num_trial, nm))
    proba_recovery_music = np.zeros(shape=(n_experiments,num_trial, nm))

    # Loop over the trials
    for trial in range(num_trial):
        print("{} th simulation".format(trial + 1))

        # Seed number
        prng = np.random.RandomState(seed=trial)

        for ll in range(nm):
            # Current measurement ratio
            m = list_m[ll]

            # Generate Gaussian random matrix
            A = np.random.randn(m, n) * np.sqrt(1 / m)

            # Signal matrix
            X = np.zeros(shape=(n, N))

            # Known support
            row_supp = np.random.permutation(n)
            pks = row_supp[:n_pks]
            for nn in range(n_experiments):
                if nn == 1:
                    X[pks] = 1
                else:
                    X[pks] = np.random.randn(n_pks, N).copy()

                # Complement of the known support
                J1 = row_supp[k-n_pks:k]
                X[J1] = np.random.randn(len(J1), N).copy()
                supp_X = np.union1d(pks, J1).astype(int)
                s = np.linalg.matrix_rank(X)

                # Measurements
                Y = np.matmul(A, X)
                s = np.linalg.matrix_rank(Y)

                # MMV signal recovery with MUSIC
                music_mmv = music.MUSIC(measurements=Y, A=A, k=len(J1) + len(pks), rank=s)
                _, supp_X_rec = music_mmv.solve()

                # MMV signal recovery with MUSIC-PKS
                music_mmv = music.MUSIC(measurements=Y, A=A, k=len(J1) + len(pks), rank=s, pks=pks)
                _, supp_X_rec_pks = music_mmv.solve()

                # Support recovery
                proba_recovery_music[nn, trial, ll] = set(supp_X_rec) == set(supp_X)

                # Support recovery
                proba_recovery_music_pks[nn, trial, ll] = set(supp_X_rec_pks) == set(supp_X)



    # Create the dictionary containing the results
    result_dict = {
        'proba_recovery_music' : proba_recovery_music,
        'proba_recovery_music_pks' : proba_recovery_music_pks,
        'measurements': list_m,
        'sparsity': k,
        'n': n,
        'N': N,
        'rank': s,
        'num-trial': num_trial,
    }
    # Check if results folder exists
    if not(os.path.exists(dir_results)):
        os.mkdir(dir_results)
    # Save
    if flag_save:
        filename_results_pickle = filename_results + '.pickle'
        pickle.dump(result_dict, open(os.path.join(dir_results, filename_results_pickle), "wb"))
else:
    filename_results_pickle = filename_results + '.pickle'
    result_dict = pickle.load(open(os.path.join(dir_results, filename_results_pickle), "rb" ))


# Plot the results

if flag_plot:
    # Get the values from the pickle file
    list_m = result_dict['measurements']
    proba_recovery_music = result_dict['proba_recovery_music']
    proba_recovery_music_pks = result_dict['proba_recovery_music_pks']

    # Max values for plots
    m_min = list_m[0]
    m_max = list_m[-1]
    y_min = 0
    y_max = 1.2

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Limit the range of the plot to only where the data is.
    ax.set_ylim(0, 1.05)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")

    # Additional parameters of the plots
    labelsize = 14
    legendsize = 14
    ticksize = 14
    dpi_plot = 300
    tableau20 = ut.tableau20_colors()
    alpha = 0.8

    # y axis
    for y in np.arange(y_min, y_max, 0.2):
        plt.plot(range(m_min, m_max), [y] * len(range(m_min, m_max)), "--", lw=0.5, color="black", alpha=0.3)

    plt.plot(list_m, np.mean(proba_recovery_music[0], axis=0).squeeze(), color=tableau20[0], lw=2,
             marker='s', label='MUSIC - Exp. 1', alpha=alpha)
    plt.plot(list_m, np.mean(proba_recovery_music_pks[0], axis=0).squeeze(), color=tableau20[4], lw=2,
             marker='s', label='MUSIC-PKS - Exp. 1', alpha=alpha)
    plt.plot(list_m, np.mean(proba_recovery_music[1], axis=0).squeeze(), color=tableau20[0], lw=2,
             marker='p', label='MUSIC - Exp. 2', alpha=alpha)
    plt.plot(list_m, np.mean(proba_recovery_music_pks[1], axis=0).squeeze(), color=tableau20[4], lw=2,
             marker='p', label='MUSIC-PKS - Exp. 2', alpha=alpha)
    ax.set_xlabel('m', fontsize=labelsize)
    ax.set_ylabel('Recovery Probability', fontsize=labelsize)
    plt.legend(fontsize=legendsize)

    if flag_save:
        fig_name = filename_results + '.pdf'
        plt.savefig(fig_name, bbox_inches='tight', dpi=dpi_plot, transparent=True)

    plt.show()
