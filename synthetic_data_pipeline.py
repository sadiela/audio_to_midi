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

def midi_to_wav(midi_path,wav_path):
    """ Convert MIDI file to a .wav file"""
    # using 44100 Hz sample rate (maybe more than needed?) 
    fs = FluidSynth(sound_font=SOUNDFONT_PATH)
    print("CONVERTING")
    fs.midi_to_audio(midi_path, wav_path)
    
# can parallelize these conversions since I have 8 CPU cores
