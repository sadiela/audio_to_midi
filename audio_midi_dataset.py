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

    def __getitem__(self, index):
        # MELSPECTROGRAMS
        M_db = calc_mel_spec(audio_file = self.audio_dir + self.audio_file_list[index])
        # LOAD MIDI
        midi = pretty_midi.PrettyMIDI(self.midi_dir + self.audio_file_list[index][:-3] + 'mid')
        midi_seqs = pretty_midi_to_seq_chunks(midi)

        empty_section_idxs = np.where(midi_seqs[1,:] == 0)[0]
        M_db_clean = np.delete(M_db, empty_section_idxs, axis=1)
        midi_seqs_clean = np.delete(midi_seqs, empty_section_idxs, axis=1)

        if midi_seqs_clean.shape[1] == 0:
            return None
        
        #chosen_chunk_idx = random.randint(0, midi_seqs_clean.shape[1]-1) # for now...
  
        return torch.tensor(M_db_clean), torch.tensor(midi_seqs_clean) #torch.tensor(M_db_clean[:,[chosen_chunk_idx],:]), torch.tensor(midi_seqs_clean[:, [chosen_chunk_idx]])

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

  print("FULL SPEC SHAPE:", full_spec_list.shape)
  print("FULL MIDI SHAPE:", full_midi_list.shape)

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
