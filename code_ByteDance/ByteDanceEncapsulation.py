# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:34:29 2021

@author: amarmore
"""

import numpy as np
import os
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import torch.cuda

path_maps = "C:/Users/amarmore/Desktop/Audio samples/MAPS"
#path_outputs = f"{path_maps}/outputs_ByteDance"
path_outputs = "C:/Users/amarmore/Desktop/Projects/PhD side projects/CNMF Haoran/Neural Nets baselines/ByteDance/outputs"

def run_estimates_file(file_path, piano_name):
    print(file_path)
    (audio, _) = load_audio(file_path, sr=sample_rate, mono=True)

    # Transcriptor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriptor = PianoTranscription(device=device)    # 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    file_name = file_path.split("/")[-1]
    transcribed_dict = transcriptor.transcribe(audio, f"{path_outputs}/{piano_name}/{file_name.replace('.wav', '')}.mid")

def run_estimates_piano(piano_name):
    for file in os.listdir(f"{path_maps}/{piano_name}/MUS"):
        if "wav" in file:
            file_path = f"{path_maps}/{piano_name}/MUS/{file}"
            run_estimates_file(file_path, piano_name)
    print("Done.")


def load_ref_in_arrays(ref_path, time_limit = 30):
    ref_intervals = []
    ref_pitches = []

    with open(ref_path) as f:
        truth_lines = f.readlines()[1:] # To discard the title/legend in ground truth

    for lines_index in range(len(truth_lines)):
        # Creates a list with the line of the reference, splitted on tabulations
        line_to_array = (truth_lines[lines_index].replace("\n", "")).split("\t")
        if line_to_array != [""]:
            if (time_limit == None) or ((time_limit != None) and (float(line_to_array[0]) < time_limit)): # if onset > time_limit (note outside of the cropped excerpt)
                ref_intervals.append([float(line_to_array[0]), float(line_to_array[1])])
                pitch = float(line_to_array[2])
                ref_pitches.append(pitch)
    return np.array(ref_intervals), np.array(ref_pitches)