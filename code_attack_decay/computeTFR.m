function X = computeTFR(inputFile) 

% read spectrogram factors
setGlobalDM();
global window noverlap nfft

% read audio
[x,fs] = audioread(inputFile);
x = mean(x,2);

% cut audios and add zeros at the end
global T
x = [x(1:min(T*fs,length(x))); zeros(noverlap,1)];

% spectrogram
S = spectrogram(x,window,noverlap,nfft,fs);
X = abs(S);

% smoothing
X = medfilt1(X,5);