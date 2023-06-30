import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle 
import os
import pretty_midi 
import numpy as np
from spectrograms import *
from midi_vocabulary import *

class MidiDataset(Dataset):
    """Midi dataset."""

    def __init__(self, audio_file_dir, midi_file_dir, l=256):
        """
        Args:
            audio_file_dir (string): Path to the wav file directory
            midi_file_dir: Path to midi file directory
        """
        self.audio_dir = audio_file_dir
        self.midi_dir = midi_file_dir
        self.audio_file_list = os.listdir(self.audio_dir) # WILL END IN .wav!!!
        self.audio_paths = [ Path(audio_file_dir) / file for file in self.audio_file_list]
        # f[:-3]+'wav'
        #self.midi_paths = [ Path(midi_file_dir) / str(f[:-3]+'wav') for file in self.audio_file_list]

    def __getitem__(self, index):
        # MELSPECTROGRAMS
        print(self.audio_dir + self.audio_file_list[index], self.midi_dir + self.audio_file_list[index][:-3] + 'mid')
        # load raw audio waveform
        y, sr = librosa.load(self.audio_dir + self.audio_file_list[index], sr=SAMPLE_RATE)
        # split into non-overlapping segments (~4 secs long)
        y = split_audio(y,SEG_LENGTH)
        # convert to melspectrograms
        M =  librosa.feature.melspectrogram(y=y, sr=sr, 
              hop_length=HOP_WIDTH, 
              n_fft=FFT_SIZE, 
              n_mels=NUM_MEL_BINS, 
              fmin=MEL_LO_HZ, fmax=7600.0)
        # transpose to be SEQ_LEN x BATCH_SIZE x EMBED_DIM
        M_transposed = np.transpose(M, (2, 0, 1)) # append EOS TO THE END OF EACH SEQUENCE!
        # logscale magnitudes
        M_db = librosa.power_to_db(M_transposed, ref=np.max)

        # LOAD MIDI
        midi = pretty_midi.PrettyMIDI(self.midi_dir + self.audio_file_list[index][:-3] + 'mid')
        array_seqs = pretty_midi_to_seq_chunks(midi)
  
        return M_db, array_seqs


    def __getname__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)


# FIX COLLATE
def collate_fn(data, collate_shuffle=True, batch_size=128): # I think this should still work
  # data is a list of 2d tensors; concatenate and shuffle all list items
  data = list(filter(lambda x: x is not None, data))
  #print(len(data))
  full_list = torch.cat(data, 0) # concatenate all data along the first axis
  if collate_shuffle:
    idx = torch.randperm(full_list.shape[0])
    return  full_list[idx].view(full_list.size())[:,batch_size]
  else:
    return full_list[:,batch_size]