#%%
import numpy as np
import matlab
import matlab.engine
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import re
import STFT


def H_initialization(mag):
    # find the largest norm L1 column in spectrogram
    ncol = np.shape(mag)[1]
    list_norm = [np.linalg.norm(mag[:, j], ord=1) for j in range(ncol)]
    index = list_norm.index(max(list_norm))
    H0 = np.array([[1e-10] * ncol])
    H0[:,index] = 1
    # return a one dimension array H0 with 1 in position where L1 norm column in STFT matrix is maximal and epsilon elsewhere
    return H0


def attack_decay_template(path, rank=1, H_init=True, note_intensity='M'):

    files = os.listdir(path)
    list_files_wav = []
    for it_files in files:
        if it_files.split(".")[-1] == "wav" and it_files.split("_")[-4] == note_intensity:
            list_files_wav.append(it_files)

    if len(list_files_wav) == 0:
        raise NotImplementedError("Empty list of songs.")
    Dictionary_decay = {}
    Dictionary_attack ={}
    Dictionary_alpha = {}
    Dictionary_P = {}
    Dictionary_results = {}
    engine = matlab.engine.start_matlab()

    #first round training (fixed H)
    print('first round')
    for name in list_files_wav:
        f = path + "/" + name
        midi = re.search(r'(?<=M)\d+', name).group(0)
        print('processing note MIDI: ', midi)
        stft = STFT.STFT(f, model_AD=True)
        mag = stft.get_magnitude_spectrogram()
        nf = np.shape(mag)[0]
        # we remove the column if all elements in that column < 1e-10
        columnlist = []
        for i in range(np.shape(mag)[1]):
            if (mag[:, i] < 1e-10).all():
                columnlist.append(i)
        mag = np.delete(mag, columnlist, axis=1)

        if H_init:
            H0 = H_initialization(mag)

        mag = matlab.double(mag.tolist())
        H0 = matlab.double(H0.tolist())
        parameters = engine.struct('R', rank, 'update', matlab.int32([1,1,1,0,1]), 'sparsity', matlab.double([1, 1.04]),
                                   'threshold', -30)
        # parameters.R = 88; % 88
        # parameters.update = [0, 0, 0, 1, 0]; % update flags for [W, TS, a, H, pattern]
        # parameters.sparsity = [1 1.04]; % annealing sparsity
        # parameters.threshold = -30  (not useful here)
        Template = matlab.double()
        initialisation = engine.setInitialisation(Template, mag, H0, parameters)
        result = engine.convNMFT(mag, initialisation)
        Dictionary_P[int(midi)] = np.asarray(result['pattern'])
        Dictionary_results[int(midi)] = result

    pattern = np.array(list(Dictionary_P.values()))
    pattern_average = np.mean(pattern, 0).flatten()
    # normalization of pattern
    pattern_average = pattern_average/max(pattern_average)
    pattern_average_matlab = matlab.double(pattern_average.tolist())

    # second round training (fixed H and P)
    print('second round')
    for name in list_files_wav:
        f = path + "/" + name
        midi = re.search(r'(?<=M)\d+', name).group(0)
        print('MIDI: ', midi)
        stft = STFT.STFT(f, model_AD= True)
        mag = stft.get_magnitude_spectrogram()

        # we remove the column if all elements in that column < 1e-10
        columnlist = []
        for i in range(np.shape(mag)[1]):
            if (mag[:, i] < 1e-10).all():
                columnlist.append(i)
        mag = np.delete(mag, columnlist, axis=1)
        print('mag shape: ',np.shape(mag))

        if H_init:
            H0 = H_initialization(mag)

        mag = matlab.double(mag.tolist())
        H0 = matlab.double(H0.tolist())
        result = Dictionary_results[int(midi)]
        parameters = engine.struct('R', rank, 'update', matlab.int32([1, 1, 1, 0, 0]), 'sparsity',
                                   matlab.double([1, 1.04]), 'threshold', -30)
        # continue using the template from first round training
        Template = engine.struct('TS', result['TS'], 'W', result['W'], 'a', result['a'], 'pattern', pattern_average_matlab)
        initialisation = engine.setInitialisation(Template, mag, H0, parameters)
        result = engine.convNMFT(mag, initialisation)

        Dictionary_attack[int(midi)] = np.asarray(result['TS'])
        Dictionary_decay[int(midi)] = np.asarray(result['W'])
        Dictionary_alpha[int(midi)] = np.asarray(result['a'])

    # build dictionary
    mat_att = np.zeros(shape=(nf, 88))
    mat_decay = np.zeros(shape=(nf, 88))
    alpha = np.zeros(88)
    for i in range(88):
        mat_att[:, i] = Dictionary_attack[i + 21].flatten()
        mat_decay[:, i] = Dictionary_decay[i + 21].flatten()
        alpha[i] = Dictionary_alpha[i + 21]
    engine.exit()
    return mat_att, mat_decay, alpha, pattern_average, pattern


def dictionary_normalization(dict_MM1, beta, type, name, plot=False):
    dict_MM = np.copy(dict_MM1)
    if type == 1:
        # normalization by Frobenius norm
        for i in range(np.shape(dict_MM1)[1]):
            dict_MM[:,i] = dict_MM1[:,i]/ np.linalg.norm(dict_MM1[:,i], ord=2)

    elif type == 2:
        # normalization by norm beta of column i
        for i in range(np.shape(dict_MM1)[1]):
            s = (sum(dict_MM1[:,i]**beta))**(1/beta)
            dict_MM[:,i] = dict_MM1[:,i]/s

    # plot result MM
    if plot:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        caxes = axes.matshow(np.log(dict_MM), interpolation='nearest', aspect='auto', vmin=-12, vmax=0)
        figure.colorbar(caxes)
        xmajorLocator = MultipleLocator(5)
        axes.xaxis.set_major_locator(xmajorLocator)
        axes.set_xticklabels(range(16, 111, 5))
        axes.set_title('Dictionary'+name, fontsize=14, color='black')
        axes.xaxis.set_ticks_position('bottom')
        axes.set_xlabel('note in MIDI format')
        axes.set_ylabel('frequency')
        axes.xaxis.set_tick_params(rotation=45, labelsize=8)
        plt.show()

    return dict_MM


def semi_supervised_transcribe(path, W_attack, W_decay, alpha, pattern, time_limit=30, H0=None, plot=False):
    stft = STFT.STFT(path, time=time_limit, model_AD=True)
    X = stft.get_magnitude_spectrogram()
    engine = matlab.engine.start_matlab()
    # we remove firstly the columns whose contents are less than 1e-10
    # columnlist = []
    # for i in range(np.shape(X)[1]):
    #     if (X[:, i] < 1e-10).any():
    #         columnlist.append(i)
    print('shape of preprocessing matrix', np.shape(X))

    parameters = engine.struct('R', 88, 'update', matlab.int32([0, 0, 0, 1, 0]), 'sparsity', matlab.double([1, 1.04]),
                               'threshold', -30)
    mag = matlab.double(X.tolist())
    Template = engine.struct('TS', matlab.double(W_attack.tolist()), 'W', matlab.double(W_decay.tolist()),
                             'a', matlab.double(alpha.tolist()), 'pattern', matlab.double(pattern.tolist()))
    if H0 is not None:
        H_init = matlab.double(H0.tolist())
    else:
        H_init = matlab.double()
    initialisation = engine.setInitialisation(Template, mag, H_init, parameters)
    result = engine.convNMFT(mag, initialisation)
    H = np.asarray(result['H'])
    Ha = np.asarray(result['Ha'])

    if plot:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.matshow(Ha, interpolation='nearest', aspect='auto')
        axes.set_title('activation matrix H', fontsize=14, color='black')
        plt.show()

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.matshow(H, interpolation='nearest', aspect='auto')
        axes.set_title('activation matrix H', fontsize=14, color='black')
        plt.show()
    return H, Ha

