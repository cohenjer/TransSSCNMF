# Codes and Supplementary Materials for Semi-supervised Convolutive NMF for Automatic Music Transcription
--------
## Contents

This repository contains complements to our paper available [here](todo) (link to be updated):

- A complement on the link between Attack Decay and CNMF, in Supplementary Material 4 (pdf and markdown).

- Scripts to perform CNMF, based on the MM algorithm proposed by [1].
  - convolutive_MM.py: the naive MM algorithm for CNMF.
  - beta_divergence.py: the beta_divergence loss.

- Jupyter notebooks and parsers to run the experiments from the paper.
  - **Supplementary material 1,2,3,4,5: run all experiments from paper**, given precomputed templates W and activations H.
  - Computation_time.ipynb: performs training and testing while measuring time.
  - launch_H_learning.py: start the test phase given a set of templates.
  - launch_W_learning.py: start the training stage to obtain templates.
  - script_igrida_learn.py: perform the learning phase.
  - script_igrida_trancrive.py: perform the test phase.

- Scripts to perform transcription (pre/post-processing and parsing)
  - compare_pianos_script.py: calls transcribe_factorization and stores results.
  - evaluate_transcription.py: evaluates results with mir_eval.
  - STFT.py: performs STFT using scipy.
  - transcribe_factorization.py: does the heavy-lifting of post-processing.

## Installation and Usage
A few dependencies are requires, which you can install with
``pip install -r req.txt``
All codes should be run from within this repository.

The notebooks ``Supplementary material xxx`` contain information on how to use the code provided here. If you want to reproduce the experiments but do not have the MAPS dataset, or do not want to wait several days for your computer to produce all the patterns and activations, you can download our templates and activations from [this link](todo) (todo, will be done later).

To produce Attack Decay patterns, some matlab code is available at `https://code.soundsoftware.ac.uk/projects/decay-model-for-piano-transcription/repository/show/Supplement/code`.

To use, please cite our CNMF manuscript as follows:
xxx Semi-supervised Convolutive NMF for Automatic Music Transcription (todo)

[1] D. Fagot et. al., Majorization-minimization algorithms for convolutive NMF with beta-divergence, ICASSP 2019.
