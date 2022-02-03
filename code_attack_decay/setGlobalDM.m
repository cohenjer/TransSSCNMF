function setGlobalDM()

% spectrogram factors 
global window noverlap frame nfft fs
window = 4096;
frame = 882;
noverlap = window-frame;
nfft = 8192;
fs = 44100;

global Wt Tt T Tmax
Wt = 0; % the number of frames of the harmonic part behind the onset
Tt = floor(window/frame); % the spectral bluring length
T = 30; % the cutoff length in second
Tmax = T/(frame/fs); % the maximum duration of a note