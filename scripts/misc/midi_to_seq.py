import pretty_midi
import os
import midi_vocabulary
import numpy as np

# create a script that converts all of the midi files in lmd tracks to numpy sequences

folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'] 
directory = "/projectnb/textconv/sadiela/audio_to_midi/lmd_tracks"
saveDirectory = "/projectnb/textconv/dcliu3/audio_to_midi/lmd_tracks_seqs"
for folder in folder_extensions:
    # if folder exists in saveDirectory, then skip, otherwise create
    if not os.path.exists(os.path.join(saveDirectory,folder)):
        os.makedirs(os.path.join(saveDirectory,folder))
    for file in os.listdir(os.path.join(directory,folder)):
        if file.endswith(".mid"):
            tempMid = pretty_midi.PrettyMIDI(os.path.join(directory,folder,file))
            tempMidData = midi_vocabulary.pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid)
            # save tempMidData in the correct folder in saveDirectory
            np.save(os.path.join(saveDirectory,folder,file[:-4],".npy"),tempMidData)
            # break
            
            # break
    # break

