import pretty_midi
import yaml
from midi2audio import FluidSynth
import os
from pathlib import Path
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm
from pathlib import Path
from numpy import ndarray
import os
import sys

SOUNDFONT_PATH = "/Users/sadiela/Documents/phd/research/music/sound_fonts/GeneralUser_GS_v1.471.sf2"

def sep_and_crop(midi_directory, target_directory):
    # if folder doesn't exist, create it
    if not os.path.isdir(str(target_directory)):
        print("creating target directory")
        os.makedirs(str(target_directory))
# takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    num_tracks = 0 
    num_errors = 0
    print("NUMBER OF FILES:", len(file_list))
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
            #print("ERROR!", e)
            num_errors += 1
            pass
    print("NUMBER OF TRACKS:", num_tracks)
    print("NUMBER ERRORS:", num_errors)


def midi_to_wav(midi_path,wav_path):
        # using the default sound font in 44100 Hz sample rate
        fs = FluidSynth(sound_font=SOUNDFONT_PATH)
        fs.midi_to_audio(midi_path, wav_path)

def midis_to_wavs(midi_dir, wav_dir=None):
    if wav_dir == None: 
        wav_dir = midi_dir
    if not os.path.isdir(str(wav_dir)):
        print("creating target directory")
        os.makedirs(str(wav_dir))
    midi_dir_list = os.listdir(midi_dir)
    # skip files that have already been converted
    midi_list = [f for f in midi_dir_list if f[-3:] == 'mid' and not os.path.isfile(str(wav_dir) +'/' + f[:-3]+'wav')]
    for midi in midi_list: 
        midi_to_wav(str(midi_dir) + '/' + midi, str(wav_dir) +'/' + midi[:-3]+'wav')
    
# can parallelize these conversions since I have 8 CPU cores

if __name__ == '__main__':
    print("Converting tracks to raw audio")
    midi_stub = './lmd_full/'
    track_stub = './lmd_tracks/'
    raw_audio_stub = './raw_audio/'
    folder_extensions = ['2','3','4','5','6','7','8','9','a','b','c','d','e','f'] # '0','1',

    # separate MIDIs into tracks and crop empty starts
    '''for e in folder_extensions: 
        print("FOLDER", e)
        cur_midi_folder = midi_stub + e 
        cur_track_folder = track_stub + e 
        sep_and_crop(Path(cur_midi_folder), Path(cur_track_folder))'''

    # convert MIDIs to wavs
    for e in folder_extensions:
        cur_track_folder = track_stub + e 
        cur_raw_audio = raw_audio_stub + e
        midis_to_wavs(cur_track_folder, cur_raw_audio)
        print("FINSIHED", e)




