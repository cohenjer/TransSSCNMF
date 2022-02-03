# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:12:07 2021

"""

import sys
import script_igrida_learn as scr

if __name__ == "__main__":
    print(sys.argv[3])
    path_maps = sys.argv[3]  #"C:/Users/amarmore/Desktop/Audio samples/MAPS" #TOMODIFY

    # Parameters and paths
    if len(sys.argv) < 4:
        raise NotImplementedError("Not enough arg, to debug")

    pianos = ["AkPnBcht","AkPnBsdf","AkPnStgb","ENSTDkAm","SptkBGAm","StbgTGd2", "AkPnCGdD", "ENSTDkCl"]
    piano = sys.argv[1]
    if piano in pianos:
        piano_type = piano
    else:
        raise NotImplementedError("Wrong piano name")

    note_intensity = "M"
    itmax = 500
    path_piano_isol = "{}/{}/ISOL/NO/".format(path_maps, piano_type)

    beta = 1
    if sys.argv[2] == "all":
        for T in [5,10,20]:
            print("T: {}".format(T))
            _, _ = scr.learning_W_and_persist(path_piano_isol, beta, T, itmax=itmax, rank=1, init="L1", model_AD = True, piano_type = piano_type, note_intensity = note_intensity)
    else:
        T = int(sys.argv[2])
        _, _ = scr.learning_W_and_persist(path_piano_isol, beta, T, itmax=itmax, rank=1, init="L1", model_AD = True, piano_type = piano_type, note_intensity = note_intensity)
    print("Done")
