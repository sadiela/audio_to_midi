import os
import pretty_midi
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd

def get_track_lengths(midi_dir, extension, length_dict):
    file_list = os.listdir(midi_dir / extension)
    print("NUMBER OF FILES:", len(file_list))
    for fname in tqdm(file_list): 
        open_midi = pretty_midi.PrettyMIDI(str(midi_dir / extension / fname))
        length_dict[extension + '/' + fname] = open_midi.get_end_time()


def find_sparse_midis(midi_directory, extension, dense_tracks, sparsity_dict, sparsity_cutoff=0.25):
    num_sparse = 0
    file_list = os.listdir(midi_directory / extension)
    print("NUMBER OF FILES:", len(file_list))
    for fname in tqdm(file_list): 
        open_midi = pretty_midi.PrettyMIDI(str(midi_directory / extension / fname))
        pianoroll = open_midi.get_piano_roll()
        pianoroll_sum = np.sum(pianoroll, axis=0)
        try: 
            note_playing_percentage = np.count_nonzero(pianoroll_sum)/pianoroll_sum.shape[0]
        except ZeroDivisionError:
            continue
        sparsity_dict[e+"/"+fname] = note_playing_percentage
        if note_playing_percentage < sparsity_cutoff:
            num_sparse += 1 
        else:
            dense_tracks.append(e + "/" + fname)
    print("TOTAL NUM SPARSE MIDIS:", num_sparse)
    return len(file_list), num_sparse #dense_tracks, sparsity_dict, sparsity_scores

def create_datasets(dense_filepath):
    ### BUILD DATA LISTS
    with open(dense_filepath, 'rb') as fp:
        dense_filelist = pickle.load(fp)
    
    test_files = [f for f in dense_filelist if f[0]=='e' or f[0]=='f']
    with open('./testfiles.p', 'wb') as fp:
        pickle.dump(test_files, fp, pickle.HIGHEST_PROTOCOL)
    
    train_files = [f for f in dense_filelist if f not in test_files]
    with open('./trainfiles.p', 'wb') as fp:
        pickle.dump(train_files, fp, pickle.HIGHEST_PROTOCOL)

    print("LENGTH OF TEST AND TRAIN LISTS:", len(test_files), len(train_files))

    input("Continue...")

    random.shuffle(test_files)
    random.shuffle(train_files)

    with open('./testfiles_sm.p', 'wb') as fp:
        pickle.dump(test_files[:100], fp)
    
    with open('./testfiles_med.p', 'wb') as fp:
        pickle.dump(test_files[:1000], fp)
    
    with open('./trainfiles_sm.p', 'wb') as fp:
        pickle.dump(train_files[:1000], fp)
    
    with open('./trainfiles_med.p', 'wb') as fp:
        pickle.dump(train_files[:10000], fp)
    
if __name__ == '__main__':
    print("FIND LENGTH STATS!")

    with open('./data_lists/length_dict.p', 'rb') as fp:
        length_dict = pickle.load(fp)

    print("dict loaded")

    length_list = list(length_dict.values())
    lengths = pd.Series(length_list)


    length_list.sort()

    lengths_in_frames = [l/4 for l in length_list]

    print(lengths.describe())

    plt.hist(length_list[:1000000], bins=50) # [:900000]
    plt.title("MIDI Length Distribution")
    plt.xlabel("Length (s)")
    plt.show() 

    plt.hist(lengths_in_frames[:1000000], bins=50) # [:900000]
    plt.title("MIDI Length Distribution")
    plt.xlabel("Length (frames)")
    plt.show() 


    print("DONE")

    sys.exit(0)



    #create_datasets('./densefiles.p')

    extension = folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    dense_tracks = []
    sparsity_dict = {}
    total_num_files = 0
    total_sparse_files = 0

    length_dict = {}
    for e in folder_extensions:
        print(e)
        print("dict keys", len(length_dict.keys()))
        get_track_lengths(Path('./lmd_tracks'), e, length_dict)
    
    midi_lengths = list(length_dict.values())

    with open('./data_lists/length_dict.p', 'wb') as fp:
        pickle.dump(length_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    plt.hist(length_dict, bins=30)
    plt.title("MIDI Length Distribution")
    plt.xlabel("Length (s)")
    plt.show() 


    sys.exit(0)


    for e in folder_extensions:
        all_files, sparse_files = find_sparse_midis(Path('./lmd_tracks'), e, dense_tracks, sparsity_dict)
        total_num_files += all_files
        total_sparse_files += sparse_files
        print("LIST LENGTH", len(dense_tracks), len(list(sparsity_dict.values())))

    print("SAVE DENSE LIST AND SPARSITY DICT")
    print("total files and sparse files:", total_num_files, total_sparse_files)

    with open('./densefiles.p', 'wb') as fp:
        pickle.dump(dense_tracks, fp, pickle.HIGHEST_PROTOCOL)

    with open('./sparsity_dict.p', 'wb') as fp:
        pickle.dump(sparsity_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("INFO SAVED")

    sparsity_scores = list(sparsity_dict.values())

    plt.hist(sparsity_scores, bins=10)
    plt.show() 

    create_datasets('./densefiles.p')

    # with open('filename.pickle', 'rb') as handle:
    #    unserialized_data = pickle.load(handle)
    
    # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

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


