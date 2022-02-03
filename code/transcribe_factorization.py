# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:40:58 2019

"""

import numpy as np
import math
from scipy.signal import find_peaks

from midiutil import MIDIFile

def transcribe_activations(midi_codebook, activation, stft, threshold, sliding_window = 5, pourcentage_onset_threshold = 0.1, H_normalization = False):
    """
    Transcribe the activation in onset-offset-pitch notes, and format it for txt and midi

    Parameters
    ----------
    midi_codebook: array
        The codebook in midi, for corresponding the activation to their pitch
    activation: array
        The activations find by NMF
    stft: STFT object (see STFT.py)
        The Short Time Fourier Transform of the original signal, used for the sampling rate
    threshold: float
        The threshold for activation to be considered a note
    sliding_window: integer
        The sliding window on which to operate avergaing of activation (minimizing localized peaks misdetection)
        Default: 5
    pourcentage_onset_threshold = 0.1
        Pourcentage of the threshold to consider for resetting the onset.
        The goal is to lower the threshold for a closer onset, as the detection happens after the real onset
        Default: 0.1

    Returns
    -------
    note_tab: list
        Content of transcription_evaluation (onset, offset and midipitch) for txt format (tab with the values in that order)
    MIDI_file_output: MIDIFile
        Content of transcription_evaluation in MIDIFile format.
    """

    presence_of_a_note = False
    note_tab = []
    current_pitch = 0
    current_onset = 0
    current_offset = 0

    # Creation of a .mid file
    MIDI_file_output = MIDIFile(1)
    MIDI_file_output.addTempo(0,0,60)

    if H_normalization:
        H_max = np.amax(activation)
    else:
        H_max = 1

    for note_index in range(0, activation.shape[0]): # Looping over the notes
        # Avoiding an uncontrolled situation (boolean to True before looking at this notes)
        if presence_of_a_note:
            presence_of_a_note = False

        for time_index in range(activation[note_index].size): # Taking each time bin (discretized in 64ms windows)
            # Looking if the activation of this note over several consecutive frames is strong enough, being larger than a defined threshold
            #onsetCondition = (0.75 * activation[note_index, time_index] > Constants.NOTE_ACTIVATION_THRESHOLD) # Note detected
            minimalSustainCondition = (np.mean(activation[note_index, time_index:time_index + sliding_window]) > threshold * H_max) # Note sustained

            if minimalSustainCondition: # note detected and sustained
                if not presence_of_a_note: # If the note hasn't been detected before
                    try:
                        current_pitch = midi_codebook[note_index] # Storing the pitch of the actual note
                        for i in range(sliding_window):
                            onset_time_index = time_index + i
                            if(activation[note_index, onset_time_index] > pourcentage_onset_threshold * threshold):
                                current_onset = stft.time_bins[onset_time_index] # Storing the onset
                                presence_of_a_note = True # Note detected (for the future frames)
                                break
                    except ValueError as err:
                        # An error occured, the note is incorrect
                        print("The " + str(note_index) + " of the codebook is incorrect: " + err.args[1])
                        break

            else:
                if presence_of_a_note and stft.time_bins[time_index] > current_onset: # End of the previous note
                    current_offset = stft.time_bins[time_index]
                    note_tab.append([current_onset, current_offset, current_pitch]) # Format for the .txt
                    MIDI_file_output.addNote(0, 0, current_pitch, current_onset, current_offset - current_onset, 100) # Adding in the .mid file

                    presence_of_a_note = False # Reinitializing the detector of a note

    return note_tab, MIDI_file_output

def transcribe_activations_dynamic(midi_codebook, H, stft, threshold, sliding_window = 10, H_normalization = False):

    presence_of_a_note = False
    note_tab = []
    current_pitch = 0
    current_onset = 0
    current_offset = 0

    # Creation of a .mid file
    MIDI_file_output = MIDIFile(1)
    MIDI_file_output.addTempo(0,0,60)

    # smoothing activation matrix (moving average)
    activation = np.zeros(np.shape(H))
    for i in range(np.shape(H)[1]):
        if i-sliding_window<0 or i + sliding_window + 1>np.shape(H)[1]:
            d = min(i, np.shape(H)[1]-i)
            activation[:,i] = np.mean(H[:,i-d:i+d+1])
        else:
            activation[:,i] = np.mean(H[:,i-sliding_window:i+sliding_window+1],axis=1)

    if H_normalization:
        H_max = np.amax(activation)
    else:
        H_max = 1

    for note_index in range(0, activation.shape[0]): # Looping over the notes
        # Avoiding an uncontrolled situation (boolean to True before looking at this notes)
        if presence_of_a_note:
            presence_of_a_note = False

        for time_index in range(activation[note_index].size):
            # Looking if the activation of the note is larger than its smooth value + threshold
            minimalSustainCondition = (H[note_index, time_index] - activation[note_index, time_index] > threshold * H_max) # actived note

            if minimalSustainCondition: # note detected and sustained
                if not presence_of_a_note: # If the note hasn't been detected before
                    try:
                        current_pitch = midi_codebook[note_index] # Storing the pitch of the actual note
                        current_onset = stft.time_bins[time_index] # find the onset time
                        presence_of_a_note = True # Note detected (for the future frames)
                    except ValueError as err:
                        # An error occured, the note is incorrect
                        print("The " + str(note_index) + " of the codebook is incorrect: " + err.args[1])
                        break
            else:
                if presence_of_a_note: # End of the previous note
                    current_offset = stft.time_bins[time_index]
                    note_tab.append([current_onset, current_offset, current_pitch]) # Format for the .txt
                    MIDI_file_output.addNote(0, 0, current_pitch, current_onset, current_offset - current_onset, 100)
                    presence_of_a_note = False # Reinitializing the detector of a note

    return note_tab, MIDI_file_output

def transcribe_activations_dynamic_AD(midi_codebook, H, stft, threshold, sliding_window = 10, H_normalization = False):

    presence_of_a_note = False
    note_tab = []
    current_pitch = 0
    current_onset = 0
    current_offset = 0

    # Creation of a .mid file
    MIDI_file_output = MIDIFile(1)
    MIDI_file_output.addTempo(0,0,60)

    # smoothing activation matrix (moving average)
    activation = np.zeros(np.shape(H))
    for i in range(np.shape(H)[1]):
        if i-sliding_window<0 or i + sliding_window + 1>np.shape(H)[1]:
            d = min(i, np.shape(H)[1]-i)
            activation[:,i] = np.mean(H[:,i-d:i+d+1])
        else:
            activation[:,i] = np.mean(H[:,i-sliding_window:i+sliding_window+1],axis=1)

    if H_normalization:
        H_max = np.amax(activation)
    else:
        H_max = 1


    for note_index in range(0, activation.shape[0]): # Looping over the notes
        line = H[note_index, :]
        line[line - activation[note_index, :] < threshold * H_max] = 0
        loc, p = find_peaks(line, height=0)
        height = p["peak_heights"]
        onset_candidat = []
        if len(loc):
            dind = np.diff(loc)
            for i in range(len(dind)):
                if dind[i]<5:
                    if loc[i] != -1:
                        onset_candidat.append(int(np.around((height[i]*loc[i]+height[i+1]*loc[i+1])/(height[i]+height[i+1]))))
                        loc[i+1]=-1
                else:
                    if loc[i] != -1:
                        onset_candidat.append(loc[i])
        presence_of_a_note = False
        for time_index in onset_candidat:
            current_pitch = midi_codebook[note_index]  # Storing the pitch of the actual note
            current_onset = stft.time_bins[time_index] # find the onset time
            current_offset = stft.time_bins[time_index+1]
            note_tab.append([current_onset, current_offset, current_pitch]) # Format for the .txt
            MIDI_file_output.addNote(0, 0, current_pitch, current_onset, current_offset - current_onset, 100)

    return note_tab, MIDI_file_output

def codebook_in_midi(codebook, stft, plot = False):
    """
    Converts the frequency codebook (in Hz) in the midi scale

    Parameters
    ----------
    codebook: array
        Codebook to convert
    stft: STFT object (see STFT.py)
        Short Time Fourier Transform of the original signal, used for the sampling rate

    Returns
    -------
    f0_of_codebook: list
        List containing the fundamental frequencies of all columns of the codebook
    """

    f0_of_codebook = []
    for i in range(0,codebook.shape[1]): # Fundamental frequency estimation of each atom of the codebook
        try:
            f0_of_codebook.append(freq_to_midi(f0_atom_spectrogram(stft, codebook[:,i], plotting = plot)))
        except ValueError as err:
            f0_of_codebook.append(err.args[0])
            print("Error in the " + str(i) + "-th note-atom of the codebook: " + err.args[1])
    return f0_of_codebook

def f0_atom_spectrogram(stft, note_spec, plotting = False, pitch_min = 50, pitch_max = 5000, spectrogram_salience_threshold = 0.3):
    """
    Fundamental frequency estimation of a frequency spectrogram, found with autocorrelation of the signal

    Parameters
    ----------
    stft: STFT object (see STFT.py)
        Short Time Fourier Transform of the original signal, used for the sampling rate
    note_spec: array (column)
        Spectrogram whose fundamental frequency is to estimate
    plotting: boolean
        Indicates whether to plot the spectrogram and the autocorrelation of the signal or not
        Default: False
    pitch_min: integer
        Minimal frequency (Hz) to consider the pitch as correct
        Default: 50
    pitch_max: integer
        Maximal frequency (Hz) to consider the pitch as correct
        Default: 5000
    spectrogram_salience_threshold: float
        Pitch salience (max not in 0 / 0 value) threshold under which the spectrogram is considered incorrect
        Default: 0.3

    Returns
    -------
    fundamental_frequency: float
        Estimation of the fundamental frequency of the spectrogram, in Hz

    References
    ----------
    [1] Julien Ricard. "Towards computational morphological description of sound,"
    DEA pre-thesis research work, Universitat Pompeu Fabra, Barcelona (2004).
    """

    inverseFourier_note_spec = np.fft.irfft(note_spec)

    autocorrelation_inverseFourier_note_spec = np.correlate(inverseFourier_note_spec, inverseFourier_note_spec, mode='same')

    # Normalization of the autocorrelation
    autocorrelation_inverseFourier_note_spec = autocorrelation_inverseFourier_note_spec / np.amax(autocorrelation_inverseFourier_note_spec)

    # Keeping the positive values (symmetric so useless)
    autocorrelation_inverseFourier_note_spec = autocorrelation_inverseFourier_note_spec[int(autocorrelation_inverseFourier_note_spec.size/2)-1:]

    # Coefficient in order not to take the maximum, occuring when the signal is correlated at time 0.
    # Either the index of the min or an arbitrary value for the case where the minimal value is far away.
    # This arbitrary value has to be large enough to eliminate enough first values which are correlate to the case of 0 delay in autocorrelation
    # (can/should be discussed)
    offset = min(60, np.argmin(autocorrelation_inverseFourier_note_spec))

    if np.amax(autocorrelation_inverseFourier_note_spec[offset:]) > spectrogram_salience_threshold:
        found_pitch = stft.sampling_rate/(offset + np.argmax(autocorrelation_inverseFourier_note_spec[offset:]))
        if found_pitch < pitch_min: # A lower bound for the frequency range, must be calculated from the size of the window
            raise ValueError(-1, 'The pitch is anormally low')

        elif found_pitch > pitch_max: # An upper bound for the frequency range, see [1], p.37
            #"when the number of simultaneous pitched components is high, the smallest common periodicity tend to be high,
            #and is likely to be greater than the higher boundary of our pitch detection range"
            #(5000Hz here, higher than standard piano upper bound (4100Hz))
            raise ValueError(-1, 'The pitch is anormally high')

        else:
            return found_pitch

    else:
        raise ValueError(-1, 'This spectrogram is irrelevant') # Error of irrelevance, too low pitch salience, see [1], error catched in

# Conversion of the fundamental frequencies in MIDI integer
# https://en.wikipedia.org/wiki/MIDI_tuning_standard#Frequency_values
def freq_to_midi(frequency):
    """
    Returns the frequency (Hz) in the MIDI scale

    Parameters
    ----------
    frequency: float
        Frequency in Hertz

    Returns
    -------
    midi_f0: integer
        Frequency in MIDI scale
    """
    return int(round(69+ 12 * math.log(frequency/440,2)))

def midi_to_freq(midi_freq):
    """
    Returns the MIDI frequency in Hertz

    Parameters
    ----------
    midi_freq: integer
        Frequency in MIDI scale

    Returns
    -------
    frequency: float
        Frequency in Hertz
    """
    return 440 * 2**((midi_freq - 69)/12)
