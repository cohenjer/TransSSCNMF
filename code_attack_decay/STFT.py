# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 17:33:39 2019

@author: amarmore
"""
import soundfile as sf
import numpy as np

from scipy import signal

class STFT:
    """ A class containing the stft coefficients and important values related to the STFT of a signal, channelwise """

    def __init__(self, path, time = None, channel='Sum', temporal_frame_size = 64/1000, model_AD = False):
        """
        STFT of a temporal signal, given a path

        Parameters
        ----------
        path: String
            Path of the signal to evaluate
        time: None or integer
            Time value (in seconds) to crop the song:
            allows to evaluate the excerpt of the sample from 0 to time seconds.
            Set to None if the entire sample should be evaluated
            Default: None
        channel: integer
            Channel of the signal on which to perform STFT
        temporal_frame_size: float
            Size of the window to perfom STFT
            Default: 0.064 (64ms)

        Attributes
        ----------
        time_bins: array
            Time bins of the STFT
        freq_bins: array
            Frequency bins of the STFT
        sampling_rate: float
            Sampling rate of the STFT
        stft_coefficients: array
            Complex coefficiens of the STFT
        """
        # For now, this function returns the stft of only one channel
        the_signal, sampling_rate_local = sf.read(path)

        if time != None:
            the_signal = the_signal[0:time*sampling_rate_local, :]

        if channel == 'Sum':
            the_signal = the_signal[:, 0] + the_signal[:, 1]
        elif channel == 'Average':
            the_signal = (the_signal[:, 0] + the_signal[:, 1])/2
        else:
            the_signal = the_signal[:, channel]

        if model_AD:
            frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
                                                         nperseg=4096,
                                                         nfft=8192, noverlap=4096 - 882)
        else:
            frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
                                                         nperseg=int(sampling_rate_local*temporal_frame_size),
                                                         nfft=int(sampling_rate_local*temporal_frame_size))
        self.time_bins = time_atoms
        self.freq_bins = frequencies
        self.sampling_rate = sampling_rate_local
        self.stft_coefficients = coeff

    def get_magnitude_spectrogram(self, threshold = None):
        """
        Computes the magnitude spectrogram of the STFT

        Parameters
        ----------
        self: the STFT

        threshold: float
            Threshold under which values will be set to zero, for denoizing

        Returns
        -------
        spec: array
            Magnitude Spectrogram of the STFT: array of the magnitudes of the STFT complex coefficients
        """

        if threshold == None:
            return np.abs(self.stft_coefficients)
        else:
            spec = np.abs(self.stft_coefficients)
            spec[spec < threshold] = 0

            # Other version, potentially helpful
            #spec = np.where(spec < np.percentile(spec, 99), 0, spec) # Forcing saprsity by keeping only the highest values

            return spec

    def get_power_spectrogram(self, threshold = None):
        """
        Computes the power spectrogram of the STFT

        Parameters
        ----------
        self: the STFT

        threshold: float
            Threshold under which values will be set to zero, for denoizing

        Returns
        -------
        spec: array
            Power Spectrogram of the STFT: array of the squared magnitudes of the STFT complex coefficients
        """

        if threshold == None:
            return np.abs(self.stft_coefficients)**2
        else:
            spec = np.abs(self.stft_coefficients)**2
            spec_zero = spec[spec < threshold] = 0
            return spec_zero

    """def get_log_spectrogram(self, threshold = None):
        log_coefficients = 20*np.log10(np.abs(self.stft_coefficients) + 10e-10) # Avoid the case of -infinity in the log with 0 value
        return preprocessing.minmax_scale(log_coefficients, feature_range=(0, 100)) # Rescalling values (for nonnegativity)"""

class stereo_STFT(STFT):
    """
    A class containing the stft coefficients of both channels in the same column and important values related to the STFT of a signal
    Inherits from STFT: the difference is that the STFT coefficients are the one of both channels, concatenated.
    """

    def __init__(self, path, time = None, temporal_frame_size = 64/1000):
        """
        STFT of a temporal signal, given a path, performed on both channels, concatenated

        Parameters
        ----------
        path: String
            Path of the signal to evaluate
        time: None or integer
            Time value (in seconds) to crop the song:
            allows to evaluate the excerpt of the sample from 0 to time seconds.
            Set to None if the entire sample should be evaluated
            Default: None
        temporal_frame_size: float
            Size of the window to perfom STFT
            Default: 0.064 (64ms)

        Attributes
        ----------
        time_bins: array
            Time bins of the STFT
        freq_bins: array
            Frequency bins of the STFT
        sampling_rate: float
            Sampling rate of the STFT
        stft_coefficients: array
            Complex coefficiens of the STFT on both channels, concatenated
        """

        the_signal, sampling_rate_local = sf.read(path)
        if time != None:
            the_signal = the_signal[0:time*sampling_rate_local,:]

        frequencies, time_atoms, left_stft =  signal.stft(the_signal[:,0], fs=sampling_rate_local, nperseg = int(sampling_rate_local*temporal_frame_size), nfft=int(sampling_rate_local*temporal_frame_size))
        frequencies, time_atoms, right_stft =  signal.stft(the_signal[:,1], fs=sampling_rate_local, nperseg = int(sampling_rate_local*temporal_frame_size), nfft=int(sampling_rate_local*temporal_frame_size))

        freq_range = np.arange(0, 2*len(frequencies))

        stft = np.append(left_stft, right_stft, axis=0)

        self.time_bins = time_atoms
        self.freq_bins = freq_range
        self.sampling_rate = sampling_rate_local
        self.stft_coefficients = stft
