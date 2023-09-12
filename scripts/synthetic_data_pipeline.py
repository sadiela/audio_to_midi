import os
import time
import pretty_midi
from tqdm import tqdm
#from numpy import ndarray
import sys
import multiprocessing
from midi_utility import *
import pickle

# PICKLE SYNTAX
''' with open(fname, 'wb') as fp:
    pickle.dump(data, fp)

with open(fname, 'rb') as fp:
    n_list = pickle.load(fp)'''


def sep_and_crop(midi_directory, target_directory):
    # separate midis into individual tracks and crop out empty beginnings. 
    # if folder doesn't exist, create it
    if not os.path.isdir(str(target_directory)):
        print("creating target directory")
        os.makedirs(str(target_directory))
    # takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    num_tracks = 0 
    num_errors = 0
    for file in tqdm(file_list): 
        try:
            open_midi = pretty_midi.PrettyMIDI(str(midi_directory / file))
            bpm = open_midi.get_tempo_changes()[1][0]
            for i, instrument in enumerate(open_midi.instruments): 
                cur_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm) # define new midi object WITH PROPER TEMPO!!!
                cur_inst = pretty_midi.Instrument(program=1) # create new midi instrument
                # copy notes from instrument to cur 
                if not instrument.is_drum:
                    num_tracks += 1
                    start_time = instrument.notes[0].start
                    for note in instrument.notes:
                        shifted_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start-start_time, end=note.end-start_time)
                        cur_inst.notes.append(shifted_note)
                    # save cur as a new midi file
                    cur_midi.instruments.append(cur_inst)
                    if not os.path.exists(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')):
                        cur_midi.write(str(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')))
                    else:
                        print(file.split('.')[0] + '_'+ str(i) + '.mid EXISTS!')
        except Exception as e:
            num_errors += 1
            pass
    print("NUMBER OF TRACKS:", num_tracks)
    print("NUMBER ERRORS:", num_errors)

def chunk_list(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]

def midis_to_wavs_multi(midi_list, midi_dir, wav_dir=None, num_processes=1): # 60!!!
    # split into chunks:
    midi_chunks = list(chunk_list(midi_list, num_processes))

    # create a list of processes
    print("Starting processes:", len(midi_chunks))
    print(midi_dir, wav_dir)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=midis_to_wavs, args=(str(midi_dir), str(wav_dir),midi_chunks[i]))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

if __name__ == '__main__':
    data_list_path = '/Users/sadiela/Documents/phd/research/music/audio_to_midi/data_lists/chunked_datalists/trainfiles_1.p'
    midi_dir = '/Users/sadiela/Documents/phd/research/music/audio_to_midi/lmd_tracks/'
    raw_audio_dir = '/Users/sadiela/Documents/phd/research/music/audio_to_midi/scc_audio/'

    with open(data_list_path, 'rb') as fp:
        data_list = pickle.load(fp)

    # create subdirectories for raw audio directory (RUN ONCE THEN YOU CAN COMMENT THIS OUT)
    folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'] 
    for f in folder_extensions:
        os.makedirs(raw_audio_dir + f, mode=0o777, exist_ok=True)

    # you can use more than 4 processes depending on how many cores your computer has
    midis_to_wavs_multi(data_list,midi_dir=midi_dir, wav_dir=raw_audio_dir, num_processes=4)

    # These should match... for whatever subdir you are looking at
    print("LIST LEN:", len(data_list), "CONVERTED FILES:", len(os.listdir(raw_audio_dir+'1')))
