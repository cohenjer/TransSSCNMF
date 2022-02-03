% test_doMultiF0

inputFile = 'MAPS_MUS-alb_se2_ENSTDkCl.wav';
outputFile = [inputFile '.txt'];
load('templates');

parameters.R = 88; % 88 pitches
parameters.update = [0,0,0,1,0]; %update flags for [W,TS,a,H,pattern]
parameters.sparsity = [1 1.04]; % annealing sparsity
parameters.threshold = -30;

doMultiF0(inputFile,outputFile,templates,parameters);