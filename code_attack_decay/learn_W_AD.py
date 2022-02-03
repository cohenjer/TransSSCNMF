import sys
import attack_decay as ad
import numpy as np

if __name__ == "__main__":
    path_maps = "/Users/haoran/Downloads/transcription_Axel/MAPS"  # TOMODIFY
    persisted_path = "../data_persisted/dictionary_AD"

    # Parameters and paths
    if len(sys.argv) < 3:
        raise NotImplementedError("Not enough arg, to debug")

    pianos_left = ["AkPnCGdD","ENSTDkCl","AkPnBcht","AkPnBsdf","AkPnStgb","ENSTDkAm","SptkBGAm","StbgTGd2"]

    try:
        piano = int(sys.argv[1])
        if piano >= 0 and piano < len(pianos_left):
            piano_type = pianos_left[piano]
        else:
            raise NotImplementedError("Not Wrong piano type, larger than the size of the number of left pianos")
    except ValueError:
        if sys.argv[1] == "akpncgdd":
            piano_type = "AkPnCGdD"
        elif sys.argv[1] == "enstdkcl":
            piano_type = "ENSTDkCl"
        else:
            raise NotImplementedError("Wrong first arg (piano type): {}".format(sys.argv[1]))

    note_intensity = sys.argv[2]
    if note_intensity not in ["F", "M", "P"]:
        raise NotImplementedError("Wrong intensity type")

    path_piano_isol = "{}/{}/ISOL/NO".format(path_maps, piano_type)

    beta = 1
    mat_att, mat_decay, alpha, pattern_average, pattern = ad.attack_decay_template(path_piano_isol)

    persisted_name_att = "attack_dict_piano_{}_beta_{}_stftAD_{}_intensity_{}".format(piano_type, beta, True, note_intensity)
    np.save("{}/{}".format(persisted_path, persisted_name_att), mat_att)

    persisted_name_decay = "decay_dict_piano_{}_beta_{}_stftAD_{}_intensity_{}".format(piano_type,beta,True,note_intensity)
    np.save("{}/{}".format(persisted_path, persisted_name_decay), mat_decay)

    persisted_name_alpha = "alpha_piano_{}_beta_{}_stftAD_{}_intensity_{}".format(piano_type, beta, True, note_intensity)
    np.save("{}/{}".format(persisted_path, persisted_name_alpha), alpha)

    persisted_name_pattern = "pattern_dict_piano_{}_beta_{}_stftAD_{}_intensity_{}".format(piano_type, beta, True,note_intensity)
    np.save("{}/{}".format(persisted_path, persisted_name_pattern), pattern_average)

    print("Done")

