import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle 
import os
import pretty_midi 
import numpy as np
from spectrograms import *
from midi_vocabulary import *
import random

class AudioMidiDataset(Dataset):
    def __init__(self, audio_file_dir, midi_file_dir):
        """
        Args:
            audio_file_dir (string): Path to the wav file directory
            midi_file_dir: Path to midi file directory
        """
        self.audio_dir = audio_file_dir
        self.midi_dir = midi_file_dir
        self.audio_file_list = os.listdir(self.audio_dir) # WILL END IN .wav!!!
        self.audio_paths = [ Path(audio_file_dir) / f for f in self.audio_file_list if f[-3:] == 'wav' ]
        self.learnable_eos = np.random.rand(1,NUM_MEL_BINS)
        # f[:-3]+'wav'
        #self.midi_paths = [ Path(midi_file_dir) / str(f[:-3]+'wav') for file in self.audio_file_list]

    def __getitem__(self, index):
        # MELSPECTROGRAMS
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
        eos_block = self.learnable_eos * np.ones((1, M_transposed.shape[1], NUM_MEL_BINS))
        M_transposed = np.append(M_transposed, np.atleast_3d(eos_block), axis=0)
        # TARGET SIZE: 512x6x512
        # logscale magnitudes
        M_db = librosa.power_to_db(M_transposed, ref=np.max)

        # LOAD MIDI
        midi = pretty_midi.PrettyMIDI(self.midi_dir + self.audio_file_list[index][:-3] + 'mid')
        midi_seqs = pretty_midi_to_seq_chunks(midi)

        empty_section_idxs = np.where(midi_seqs[0,:] == 0)[0]
        M_db_clean = np.delete(M_db, empty_section_idxs, axis=1)
        midi_seqs_clean = np.delete(midi_seqs, empty_section_idxs, axis=1)

        if midi_seqs_clean.shape[1] == 0:
            return None
        
        chosen_chunk_idx = random.randint(0, midi_seqs_clean.shape[1]-1) # for now...
  
        return torch.tensor(M_db_clean[:,[chosen_chunk_idx],:]), torch.tensor(midi_seqs_clean[:, [chosen_chunk_idx]])

    def __getname__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.audio_paths)

# FIX COLLATE
def collate_fn(data, collate_shuffle=True, batch_size=128): # I think this should still work
  # data is a list of 2d tensors; concatenate and shuffle all list items
  data = list(filter(lambda x: x is not None, data))
  specs = [item[0] for item in data]
  midis = [item[1] for item in data]
  #print(len(data))
  full_spec_list = torch.cat(specs, 1) # concatenate all data along the first axis
  full_midi_list = torch.cat(midis, 1)
  print("NEW SHAPES",full_spec_list.shape,full_midi_list.shape)

  return full_spec_list, full_midi_list

  '''if collate_shuffle:
    idx = torch.randperm(full_list.shape[0])
    return  full_list[idx].view(full_list.size())[:,batch_size]
  else:
    return full_list[:,batch_size]'''
  

if __name__ == '__main__':
    midi_dir = './small_matched_data/midi/'
    audio_dir = './small_matched_data/raw_audio/'

    dataset = AudioMidiDataset(audio_dir, midi_dir)
    print(dataset.__len__()) # only 25 files rn 
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn,shuffle=True)
    for x,y in dataloader:
      print(x.shape,"Targets",y.shape,"\n")
