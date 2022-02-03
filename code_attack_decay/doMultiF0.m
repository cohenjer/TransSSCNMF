function  []  = doMultiF0(inputFile,outputFile,templates,parameters)
% input wave file
% output txt fiel
% templates
% threshold in dB

% compute spectrogram
X = computeTFR(inputFile);

% set initialisation
initialisation = setInitialisation(templates,X,[],parameters);

% transcription
result = convNMFT(X,initialisation);

% note tracking
Note = noteTracking(X, result, parameters.threshold);

% Print output
for i=1:size(Note,1)
    Note(i,3) = 27.5*2.^((( Note(i,3)-1)*10 )/120);
end
fid=fopen(outputFile,'w');
for i=1:size(Note,1)
      fprintf(fid,'%.2f\t%.2f\t%.2f\n',Note(i,1),Note(i,2),Note(i,3));
end;
fclose(fid);
fprintf('%s','done');
fprintf('\n');