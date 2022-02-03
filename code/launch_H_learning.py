# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:12:07 2021

"""
import numpy as np
import os
import script_igrida_transcribe as scr
import sys
import time

if __name__ == "__main__":
    # Parameters and paths
    pianos = ["AkPnCGdD","ENSTDkCl","AkPnBcht","AkPnBsdf","AkPnStgb","ENSTDkAm","SptkBGAm","StbgTGd2"]

    # Jeremy
    #path_root_maps = "/home/jecohen/Travail/Toolbox/Data/Audio_Prozik/Maps/MAPS"
    # Axel
    path_root_maps = sys.argv[4]  # "C:/Users/amarmore/Desktop/Audio samples/MAPS"

    # Only for Axel (Windows problem), so you don't have to modify it (in theory)
    # backup_persisted_path = "C:/Users/amarmore/Desktop/cnmf_res"


    if len(sys.argv) < 5:
        raise NotImplementedError("Not enough arg, to debug")

    try:
        piano_W = sys.argv[1]
        if piano_W not in pianos:
            raise NotImplementedError("Wrong piano for W templates")
        piano_H = sys.argv[2]
        if piano_H not in pianos:
            raise NotImplementedError("Wrong piano type for H transcription")
    except ValueError:
        raise NotImplementedError("The piano type (1st or 2nd argument) should be a MAPS piano name, instead got: {}".format(sys.argv[1], sys.argv[2]))

    note_intensity = "M"
    beta = 1

    print(f"Piano templates learned on: {piano_W}")

    path_songs = "{}/{}/MUS".format(path_root_maps, piano_H)
    persisted_path = "../data_persisted"

    itmax_W = 500
    init = "L1"
    model_AD = True #Model AD: STFT du même type qu'Attack-Decay, devrait toujours être à True

    time_limit = 30
    itmax_H = 100
    tol = 1e-8
    files = os.listdir(path_songs)
    list_files_wav = []
    for it_files in files:
        if it_files.split(".")[-1] == "wav":
            list_files_wav.append(it_files)

    if sys.argv[3] == "all":
        for T in [5, 10, 20]:
            print(f"T: {T}")
            W_persisted_name = "conv_dict_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_{}".format(piano_W, beta, T, init, model_AD, itmax_W, note_intensity)
            try:
                dict_W = np.load("{}/{}.npy".format(persisted_path, W_persisted_name))
            except FileNotFoundError:
                raise NotImplementedError("Dictionary could not be found, to debug (probably a wrong T)")

            for a_song in list_files_wav:
                song_name = a_song.replace(".wav", "")
                print("processing piano song: {}".format(song_name))
                path_this_song = "{}/{}".format(path_songs, a_song)
                H_to_persist_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(song_name, piano_W, beta, T, init, model_AD, itmax_H, note_intensity, time_limit, tol)

                try:
                    np.load("{}/activations/{}.npy".format(persisted_path, H_to_persist_name), allow_pickle = True)
                    print("Found in loads.")
                except FileNotFoundError:
                    time_start = time.time()
                    H, n_iter, all_err = scr.semi_supervised_transcribe_cnmf(path_this_song, beta, itmax_H, tol, dict_W, time_limit = time_limit,
                                                                    H0=None, plot=False, model_AD=True, channel = "Sum")
                    print("Time: {}".format(time.time() - time_start))
                    try:
                        np.save("{}/activations/{}".format(persisted_path, H_to_persist_name), H)
                    except:
                        np.save("{}/backup/{}".format(backup_persisted_path, H_to_persist_name), H)
                        print("Saved in backup folder")
    else:
        T = int(sys.argv[3])

        W_persisted_name = "conv_dict_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_{}".format(piano_W, beta, T, init, model_AD, itmax_W, note_intensity)
        try:
            dict_W = np.load("{}/{}.npy".format(persisted_path, W_persisted_name))
        except FileNotFoundError:
            raise NotImplementedError("Dictionary could not be found, to debug (probably a wrong T)")

        for a_song in list_files_wav:
            song_name = a_song.replace(".wav", "")
            print("processing piano song: {}".format(song_name))
            path_this_song = "{}/{}".format(path_songs, a_song)
            H_to_persist_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(song_name, piano_W, beta, T, init, model_AD, itmax_H, note_intensity, time_limit, tol)

            try:
                np.load("{}/activations/{}.npy".format(persisted_path, H_to_persist_name), allow_pickle = True)
                print("Found in loads.")
            except FileNotFoundError:
                time_start = time.time()
                H, n_iter, all_err = scr.semi_supervised_transcribe_cnmf(path_this_song, beta, itmax_H, tol, dict_W, time_limit = time_limit,
                                                                H0=None, plot=False, model_AD=True, channel = "Sum")
                print("Time: {}".format(time.time() - time_start))
                try:
                    np.save("{}/activations/{}".format(persisted_path, H_to_persist_name), H)
                except:
                    np.save("{}/backup/{}".format(backup_persisted_path, H_to_persist_name), H)
                    print("Saved in backup folder")
    print("Done.")
