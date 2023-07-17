import os
import pretty_midi
from tqdm import tqdm
#from numpy import ndarray
import sys
import multiprocessing
import numpy as np
from pathlib import Path
import pickle

def find_sparse_midis(midi_directory):
    dense_tracks = []
    num_sparse = 0
    file_list = os.listdir(midi_directory)
    print("NUMBER OF FILES:", len(file_list))
    for file in tqdm(file_list): 
        try:
            open_midi = pretty_midi.PrettyMIDI(str(midi_directory / file))
            pianoroll = open_midi.get_piano_roll()
            #print("SHAPE:", pianoroll.shape)
            pianoroll_sum = np.sum(pianoroll, axis=0)
            #print("SHAPE:", pianoroll_sum.shape[0])
            note_playing_percentage = np.count_nonzero(pianoroll_sum)/pianoroll_sum.shape[0]
            #print("PLAY PERCENTAGE", note_playing_percentage)
            if note_playing_percentage < 0.2:
                num_sparse += 1 
            else:
                dense_tracks.append(file)
            #input("CONTINUE...")
        except Exception as e: 
            num_sparse += 1
    print("TOTAL NUM SPARSE MIDIS:", num_sparse)
    return dense_tracks

if __name__ == '__main__':
    print("FIND SPARSE MIDIS!")
    extension = folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    full_dense_list = []

    for e in folder_extensions:
        cur_midi_folder = Path('./lmd_tracks') / e
        print(cur_midi_folder)
        cur_dense_tracks = find_sparse_midis(cur_midi_folder)
        print(cur_dense_tracks[0], type(cur_dense_tracks[0]))
        cur_paths = [e + "/" + name for name in cur_dense_tracks]
        print(cur_paths[0])
        full_dense_list.extend(cur_paths)
        print("LIST LENGTH", len(full_dense_list))

    input("SAVE DENSE LIST")

    with open('./densefiles.p', 'wb') as fp:
            pickle.dump(full_dense_list, fp)
    


    '''for e in folder_extensions:
        cur_midi_folder = Path('./lmd_tracks') / e
        print(cur_midi_folder)
        cur_dense_tracks = find_sparse_midis(cur_midi_folder)
        with open('./densefiles_'+ e + '.p', 'wb') as fp:
            pickle.dump(cur_dense_tracks, fp)'''



'''# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('sampleList', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list'''


