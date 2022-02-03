# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:52:17 2019

"""

import numpy as np
#import mir_eval

## NB: You should use mir_eval instead, which is the reference in the domain.

def compute_statistical_rates_on_files(ground_truth_path, prediction_path, time_limit = None, onset_tolerance = 50/1000):
    """
    Compare the found transcription_evaluation (in a file) to the ground truth (in a file)
    and returns the statistical outputs (True Positives, False Positives, False Negatives) of this transcription_evaluation.
    Only the onset and the pitch are taken in account for the statistical outputs (not the offset).
    The tolerance on the onset is given by the parameter onset_tolerance

    Parameters
    ----------
    ground_truth_path: String
        The path of the transcription_evaluation reference
    prediction_path: String
        The path of the found transcription_evaluation
    time_limit: None or integer
        The time limit index, to crop the reference when only an excerpt is transcribed
        Default: None
    onset_tolerance: float
        Tolerance on the onset detection
        The found_onset will be considered correct if found_onset = ground_truth_onset +- onset_tolerance (in s)
        Default: 0.05 (50ms)

    Returns
    -------
    true_positive_rate: integer
        Number of true positives (Correctly detected notes: pitch and onset)
    false_positive_rate: integer
        Incorrectly transcribed notes (wrong pitch, wrong onset, or doesn't exit)
    false_negative_rate: integer
        Untranscribed notes (note in the ground truth, but not found in transcription_evaluation with the correct pitch and the correct onset)

    """
    pred_list = []

    #with open(Constants.RESULTS_PATH + escaper + output_name + ".txt") as f:
    with open(prediction_path) as f:
        pred_lines = f.readlines()[1:]

    for note in pred_lines:
        pred_list.append((note.replace("\n","")).split("\t"))

    truth_array = load_ref_in_array(ground_truth_path, time_limit)

    return compute_statistical_rates_on_array(truth_array, pred_list, onset_tolerance = onset_tolerance)


def compute_statistical_rates_on_array(truth_notes_array, predicted_notes_array, onset_tolerance = 50/1000, verbose = False):

    predicted_notes = predicted_notes_array.copy()
    truth_notes = truth_notes_array.copy()

    true_positive_rate = 0
    false_positive_rate = 0
    false_negative_rate = 0

    pred_nb = len(predicted_notes)
    truth_nb = len(truth_notes)

    while predicted_notes: # As it removes one element of the list at each iteration, the stop condition can be over the existence of the list
        pred_note = predicted_notes[0]
        a_detection = False
        for truth_note_index in range(len(truth_notes)):
            truth_note = truth_notes[truth_note_index]
            # Onset validation
            if float(pred_note[0]) <= float(truth_note[0]) + onset_tolerance and float(pred_note[0]) >= float(truth_note[0]) - onset_tolerance:
                # Pitch validation
                if int(pred_note[2]) == int(truth_note[2]):
                    a_detection = True
                    true_positive_rate += 1
                    truth_notes.pop(truth_note_index)
                    break
        if not a_detection: # Note hasn't been found in the ground truth
            false_positive_rate += 1
        predicted_notes.pop(0)

    false_negative_rate = len(truth_notes)

    # Raise an error if the number of evaluated notes (correct or incorrect) is false
    if 2*true_positive_rate + false_negative_rate + false_positive_rate != pred_nb + truth_nb:
        raise Exception("Incorrect number of note parsing")

    if verbose:
        print(str(truth_nb) + " notes in the reference, " + str(pred_nb) + " found in the transcription_evaluation:\n TP: " + str(true_positive_rate) + ", FN: " + str(false_positive_rate) + ", FP: " + str(false_negative_rate))

    return true_positive_rate, false_positive_rate, false_negative_rate

def load_ref_in_array(ref_path, time_limit = None):
    """
    Load the ground truth transcription_evaluation in an array, for comparing it to the found transcription_evaluation in 'compute_statistical_rates_on_array()'
    The reference needs to be a txt, and format as in MAPS (which is the dataset for which this function has been developed)

    Parameters
    ----------
    ref_path: String
        The path to the reference file (in txt)
    time_limit: None or integer
        The time limit index, to crop the reference when only an excerpt is transcribed
        Default: None

    Returns
    -------
    truth_array: list of lists
        List of all notes of the reference, format in lists containing Onsets, Offsets and Pitches, at respective indexes 0, 1, 2
    """
    truth_array = []

    with open(ref_path) as f:
        truth_lines = f.readlines()[1:] # To discard the title/legend in ground truth

    for lines_index in range(len(truth_lines)):
        # Creates a list with the line of the reference, splitted on tabulations
        if truth_lines[lines_index] != '\n':
            line_to_array = (truth_lines[lines_index].replace("\n", "")).split("\t")
            if (time_limit != None) and (float(line_to_array[0]) > time_limit): # if onset > time_limit (note outside of the cropped excerpt)
                truth_lines = truth_lines[:lines_index]
                break
            else:
                truth_array.append(line_to_array)


    return truth_array

def precision(TP, FP):
    """
    Computes the precision of the transcription_evaluation:

        Precision = True Positives / (True Positives + False Positives)

    Parameters
    ----------
    TP: integer
        Number of true positives (Correctly detected notes: pitch and onset)
    FP: integer
        Incorrectly transcribed notes (wrong pitch, wrong onset, or doesn't exit)

    Returns
    -------
    precision: float
        The precision
    """
    try:
        return TP/(TP + FP)
    except ZeroDivisionError:
        return 0

def recall(TP, FN):
    """
    Computes the recall of the transcription_evaluation:

        Recall = True Positives / (True Positives + False Negatives)

    Parameters
    ----------
    TP: integer
        Number of true positives (Correctly detected notes: pitch and onset)
    FN: integer
        Untranscribed notes (note in the ground truth, but not found in transcription_evaluation with the correct pitch and the correct onset)

    Returns
    -------
    recall: float
        The Recall
    """
    try:
        return TP/(TP + FN)
    except ZeroDivisionError:
        return 0

def accuracy(TP, FP, FN):
    """
    Computes the accuracy of the transcription_evaluation:

        Accuracy = True Positives / (True Positives + False Positives + False Negatives)

    Parameters
    ----------
    TP: integer
        Number of true positives (Correctly detected notes: pitch and onset)
    FP: integer
        Incorrectly transcribed notes (wrong pitch, wrong onset, or doesn't exit)
    FN: integer
        Untranscribed notes (note in the ground truth, but not found in transcription_evaluation with the correct pitch and the correct onset)

    Returns
    -------
    accuracy: float
        The Accuracy
    """
    try:
        return TP/(TP + FP + FN)
    except ZeroDivisionError:
        return 0

def F_measure(precision, recall):
    """
    Computes the F-measure of this transcription_evaluation:

        F-measure = 2 * precision * recall / (precision + recall)

    Parameters
    ----------
    precision: float
        The precision of this transcription_evaluation
    recall: float
        The recall of this transcription_evaluation

    Returns
    -------
    F-meaure: float
        The F-measure of this transcription_evaluation
    """
    try:
        return (2*precision*recall)/(precision + recall)
    except ZeroDivisionError:
        return 0

# A function encapsulating the usual function to calculate the scores in MIREX/papers
### Frequencies need to be in Hz !!!
# Note: useless, as it finds the same results than hand crafted functions, but this library can be useful.
# Kept commented not to forget this lib.
"""def mirex_pitch_onset_results(reference, predicted, pitch_tol = 50):
    ref = np.array(reference, float)
    est = np.array(predicted, float)
    ref_pitches = np.array(ref[:,2], int)
    est_pitches = np.array(est[:,2], int)

    (prec, rec, f_mes, useless) = mir_eval.transcription_evaluation.precision_recall_f1_overlap(ref[:,0:2], ref_pitches, est[:,0:2], est_pitches, offset_ratio = None, pitch_tolerance = pitch_tol)
    return prec, rec, f_mes"""

def print_results(TP, FP, FN):
    """
    A pretty printing of the statistical outputs, based on the True Positive, False Positive and False Negative rates
    Prints these three rates, the precision, the recall, the F-Measure and the accuracy of this transcription_evaluation.

    Parameters
    ----------
    TP: integer
        Number of true positives (Correctly detected notes: pitch and onset)
    FP: integer
        Incorrectly transcribed notes (wrong pitch, wrong onset, or doesn't exit)
    FN: integer
        Untranscribed notes (note in the ground truth, but not found in transcription_evaluation with the correct pitch and the correct onset)

    """
    print("------ Transcription scores -------")
    print("True Positive count: " + str(TP) + " (accurate notes)")
    print("False Positive count: " + str(FP) + " (detected notes but not in the reference)")
    print("False Negative count: " + str(FN)+ " (notes from the reference but undetected)")
    print("-----------------------------------")
    prec = precision(TP, FP)
    rec = recall(TP, FN)
    acc = accuracy(TP, FP, FN)
    print("Precision: " + str(prec))
    print("Recall: " + str(rec))
    print("F Measure: " + str(F_measure(prec, rec)))
    print("Accuracy: " + str(acc))
    print("-----------------------------------")
