import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import subspacemethods.music as music
import subspacemethods.greedy as sgreedy
import cs_algorithms.greedy.iht as iht
import h5py


def dft_matrix(N):
    return sp.fft(sp.eye(N))


def ctranspose(A):
    return np.conjugate(A.T)


def load_verasonics_data(hdf5_file, frame_rate):
    # Load the hdf5 file
    f = h5py.File(hdf5_file, "r")

    # Access the datasets
    rawdata_set = f['rawdata']
    probe_set = f['probe']
    angle_set = f['angles']
    excitation_set = f['tri_state_excitation_verasonics']

    # Create the sequence
    settings_attributes = ['name', 'sampling_frequency', 'transmit_frequency', 'transmit_cycles', 'demodulation_frequency', 'initial_time',
                      'mean_sound_speed', 'wavelength', 'angles']
    settings = dict()
    for attr in settings_attributes:
        if attr == 'angles':
            settings[attr] = angle_set.value
        elif attr in ['transmit_cycles']:
            settings[attr] = excitation_set.attrs.get(attr, default=None)

        else:
            settings[attr] = f.attrs.get(attr, default='None')

    # Create the probe
    probe_attributes = {'name', 'pitch_x', 'pitch_y', 'kerf_x', 'kerf_y', 'element_number_x', 'element_number_y',
                        'element_width', 'element_height', 'elevation_focus', 'bandwidth', 'impulse_cycles', 'central_frequency'
                        'impulse_window', 'impulse_wave', 'transmit_cycles'}
    probe = {}
    for attr in probe_attributes:
        if attr in ['impulse_cycles', 'impulse_window']:
            probe[attr] = probe_set.attrs.get(attr)
        else:
            probe[attr] = f.attrs.get(attr, default='None')

    # Accessing the data
    frame_number = rawdata_set.shape[0]
    ind_list = range(0, frame_number, frame_rate)
    rawdata = rawdata_set[ind_list]

    return rawdata, settings, probe


def cmux_matrix(n_channels, n_multiplex_channels, n_time_samples):
    # Number of channels per CMUX
    number_channel_per_cmux = np.round(n_channels / n_multiplex_channels).astype(int)

    # CMUX matrix
    H = []
    for kk in range(n_multiplex_channels):
        # Create a CMUX matrix
        H_s = np.zeros(shape=(n_time_samples, n_channels * n_time_samples), dtype=np.float32)
        for ll in range(number_channel_per_cmux):
            # Generate the chipping sequence
            chipping_sequence = np.random.binomial(n=1, p=0.5, size=(n_time_samples, ))
            chipping_sequence[chipping_sequence == 0] = -1

            # Put the values at the right location in the matrix
            ind_init = kk * n_time_samples * number_channel_per_cmux + ll * n_time_samples
            ind_end = kk* n_time_samples * number_channel_per_cmux + (ll+1) * n_time_samples
            H_s[:, ind_init:ind_end ] = np.diag(chipping_sequence)

        # Add the CMUX matrix to the set of all CMUX matrices
        if H == []:
            H = H_s
        else:
            H = np.concatenate((H, H_s), axis=0)

    return H


def gaussian_matrix(m, n, mean=0, stdev=1, seed=2, orthogonalize=False):
    # Generate Gaussian sensing matrix
    prng = np.random.RandomState(seed=seed)
    matrix_orth = prng.normal(loc=mean, scale=stdev, size=(n, n))

    if orthogonalize:
        matrix_orth = sp.linalg.orth(matrix_orth)
    mat = matrix_orth[0:m, :]

    return mat


def signal_matrix(n, N, rank, sparsity):
    # Signal matrix: X = U Lambda V', with rank(Lambda) = s
    U = np.random.randn(n, rank)
    U = sp.linalg.orth(U)
    Lambda = np.eye(rank, rank)
    V = np.random.randn(N, rank)
    if rank > 1:
        V = sp.linalg.orth(V)
    X = np.matmul(U, np.matmul(Lambda, V.T))

    # Mask the signal matrix to make it k-row sparse
    row_supp = np.random.permutation(n)
    row_supp = np.sort(row_supp[:sparsity])
    mask = np.zeros(shape=(n, 1))
    mask[row_supp] = 1
    X = mask * X

    return X, row_supp


def run_numerical_experiment(num_trial, n, N, list_m, k, list_s, snr, list_algorithms):
    # Sizes
    n_alg = len(list_algorithms)
    nm = len(list_m)
    ns = len(list_s)

    # Probability of recovery
    ratio_pks = [0, 0.25, 0.5, 0.75, 0.9]
    proba_recovery = np.zeros(shape=(len(ratio_pks), n_alg, num_trial, ns, nm), dtype=np.double)

    # Main loop
    for kk in range(num_trial):
        print("{} th simulation".format(kk + 1))
        for jj, algo in enumerate(list_algorithms):
            print("{} algorithm".format(algo))
            # Loop over the ranks
            for pp in range(ns):
                s = list_s[pp]
                for ll in range(nm):
                    # Current measurement ratio
                    m = list_m[ll]

                    # Generate Gaussian random matrix
                    A = np.random.randn(m, n) * np.sqrt(1 / m)

                    # Generate signal matrix
                    X, row_supp = signal_matrix(n, N, rank=s, sparsity=k)

                    # Create measurements
                    Y = np.matmul(A, X)

                    # Add noise
                    Z = np.random.randn(m, N)
                    Z *= 1 / np.linalg.norm(Z, ord='fro') * np.linalg.norm(Y, ord='fro') * 10 ** (-snr / 20)
                    Y += Z

                    for uu in range(len(ratio_pks)):
                        # Signal reconstruction - no PKS
                        T0 = row_supp[:np.round(ratio_pks[uu] * k).astype(int)]
                        X_rec, supp_X_rec = reconstruct_signal(algo, Y, A, k, s, T0)
                        proba_recovery[uu, jj, kk, pp, ll] = set(supp_X_rec) == set(row_supp)

    return proba_recovery, list_algorithms


def reconstruct_signal(algo, measurements, A, k, rank, pks):
    # Construct the reconstruction algorithm
    if algo == 'RA-ORMP':
        rec_algo = sgreedy.RAORMP(measurements=measurements, A=A, k=k, rank=rank, pks=pks, verbose='NONE')
    elif algo == 'MUSIC':
        rec_algo = music.MUSIC(measurements=measurements, A=A, k=k, rank=rank, pks=pks)
    elif algo == 'CS-MUSIC':
        rec_algo = music.CSMUSIC(measurements=measurements, A=A, k=k, rank=rank, pks=pks)
    elif algo == 'SA-MUSIC':
        rec_algo = music.SAMUSIC(measurements=measurements, A=A, k=k, rank=rank, pks=pks)
    elif algo == 'SNIHT':
        rec_algo = iht.IHT(measurements=measurements, A=A, k=k, max_iter=1000, pks=pks, acceleration='normalized', verbose='NONE')
    else:
        raise ValueError('Algorithm {} not used in the experiments'.format(algo))

    # Reconstruction
    X_rec, supp_X_rec = rec_algo.solve()

    return X_rec, supp_X_rec

def plot_kwargs(algo):
    # Create a dictionary
    plt_kwargs = {}

    # Tableau tableau20 colors for the plots
    tableau20 = tableau20_colors()

    # kwargs parameters used for the plots
    if algo == 'RA-ORMP':
        plt_kwargs['color'] = tableau20[0]
        plt_kwargs['marker'] = 's'
    elif algo == 'CS-MUSIC':
        plt_kwargs['color'] = tableau20[2]
        plt_kwargs['marker'] = 'o'
    elif algo == 'SA-MUSIC':
        plt_kwargs['color'] = tableau20[4]
        plt_kwargs['marker'] = 'p'
    elif algo == 'SNIHT':
        plt_kwargs['color'] = tableau20[6]
        plt_kwargs['marker'] = '*'
    else:
        raise ValueError('Algorithm {} not used in the experiments'.format(algo))

    # MISC parameters
    plt_kwargs['linestyle'] = '-'
    plt_kwargs['linewidth'] = 2
    plt_kwargs['alpha'] = 0.8
    plt_kwargs['label'] = algo

    # kwargs for PKS algorithms
    plt_kwargs_pks = plt_kwargs.copy()
    plt_kwargs_pks['label'] = '-'.join([plt_kwargs['label'], 'PKS'])
    plt_kwargs_pks['linestyle'] = ''.join(['-', plt_kwargs['linestyle']] )

    return plt_kwargs, plt_kwargs_pks

def plot_fig_spl(proba_recovery, proba_recoverypks, list_m, list_s, figname, list_algorithms, flag_plot=1, flag_save=0):

    # Max values for plots
    m_min = list_m[0]
    m_max = list_m[-1]
    s_min = list_s[0]
    s_max = list_s[-1]
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
    labelsize = 24
    legendsize = 18
    ticksize = 18
    dpi_plot = 300

    # Dictionary for the output filename
    filenames = {}
    filenames['a'] = 'proba_recov_pks_noiseless.pdf'
    filenames['b'] = 'proba_recov_pks_noiseless_rd.pdf'
    filenames['c'] = 'proba_recov_pks_noiseless_rank.pdf'
    filenames['d'] = 'proba_recov_pks_noisy.pdf'
    filenames['e'] = 'proba_recov_pks_noisy_rd.pdf'
    filenames['f'] = 'proba_recov_pks_noisy_rank.pdf'

    # Plots
    if figname == 'a' or figname == 'd':
        # Limits of the x-axis
        ax.set_xlim(m_min, m_max)

        # y axis
        for y in np.arange(y_min, y_max, 0.2):
            plt.plot(range(m_min, m_max), [y] * len(range(m_min, m_max)), "--", lw=0.5, color="black", alpha=0.3)

        # Color used for the plot
        tableau20 = tableau20_colors()
        alpha = 0.8

        # Plots
        plt.plot(list_m, np.mean(proba_recovery[0], axis=0).squeeze(), color=tableau20[0], lw=2,
                 marker='s', label='0%', alpha=alpha)
        plt.plot(list_m, np.mean(proba_recoverypks['25'][0], axis=0).squeeze(), color=tableau20[2], lw=2,
                 marker='o', label='25%', alpha=alpha)
        plt.plot(list_m, np.mean(proba_recoverypks['50'][0], axis=0).squeeze(), color=tableau20[4], lw=2,
                 marker='p', label='50%', alpha=alpha)
        plt.plot(list_m, np.mean(proba_recoverypks['75'][0], axis=0).squeeze(), color=tableau20[6], lw=2,
                 marker='x', label='75%', alpha=alpha)
        plt.plot(list_m, np.mean(proba_recoverypks['90'][0], axis=0).squeeze(), color=tableau20[8], lw=2,
                 marker='*', label='90%', alpha=alpha)

        # Label of the x-axis
        ax.set_xlabel('m', fontsize=labelsize)

    if figname == 'b' or figname == 'e':
        # Limits of the x-axis
        ax.set_xlim(m_min, m_max)

        # y axis
        for y in np.arange(y_min, y_max, 0.2):
            plt.plot(range(m_min, m_max), [y] * len(range(m_min, m_max)), "--", lw=0.5, color="black", alpha=0.3)

        for jj, algo in enumerate(list_algorithms):
            # Get the arguments of the plot
            plt_kwargs, plt_kwargs_pks = plot_kwargs(algo)
            # Plots
            plt.plot(list_m, np.mean(proba_recovery[jj], axis=0).squeeze(), **plt_kwargs)
            plt.plot(list_m, np.mean(proba_recoverypks[jj], axis=0).squeeze(), **plt_kwargs_pks)

        # Label of the x-axis
        ax.set_xlabel('m', fontsize=labelsize)

    elif figname == 'c' or  figname == 'f':
        # Limits of the x-axis
        ax.set_xlim(s_min, s_max)

        # y-axis
        for y in np.arange(y_min, y_max, 0.2):
            plt.plot(range(s_min, s_max), [y] * len(range(s_min, s_max)), "--", lw=0.5, color="black", alpha=0.3)

        for jj, algo in enumerate(list_algorithms):
            # Get the arguments of the plot
            plt_kwargs, plt_kwargs_pks = plot_kwargs(algo)

            # Plots
            plt.plot(list_s, np.mean(proba_recovery[jj], axis=0).squeeze(), **plt_kwargs)
            plt.plot(list_s, np.mean(proba_recoverypks[jj], axis=0).squeeze(), **plt_kwargs_pks)

        # Label of the x-acis
        ax.set_xlabel('s', fontsize=labelsize)

    # Labels and legends
    ax.set_ylabel('Recovery Probability', fontsize=labelsize)
    #plt.legend(loc='lower right', fontsize=legendsize)

    # Size of the ticks
    ax.xaxis.set_tick_params(labelsize=ticksize)
    ax.yaxis.set_tick_params(labelsize=ticksize)

    # Saving if flag_save is activated
    if flag_save:
        plt.savefig(filenames[figname], bbox_inches='tight', dpi=dpi_plot, transparent=True)

    # Plotting if flag_plot is activated
    if flag_plot:
        plt.show()


def tableau20_colors():
    # Tableau 20 colors as RGB
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20