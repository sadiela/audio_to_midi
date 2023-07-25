import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle 
import os
import pretty_midi 
import numpy as np
from spectrograms import *
from midi_vocabulary import *

MAX_BATCH=16

class AudioMidiDataset(Dataset):
    def __init__(self, dense_midis, audio_file_dir='/raw_audio/', midi_file_dir='./lmd_tracks/'):
        """
        Args:
            audio_file_dir (string): Path to the wav file directory
            midi_file_dir: Path to midi file directory
        """
        #with open(midi_pickle, 'rb') as fp:
        #    dense_midis = pickle.load(fp)
        self.dense_midis = dense_midis

        self.audio_dir = audio_file_dir
        self.midi_dir = midi_file_dir

    def __getitem__(self, index):
        # MELSPECTROGRAMS
        M_db = calc_mel_spec(audio_file = self.audio_dir + self.dense_midis[index][:-3] + 'wav')
        midi = pretty_midi.PrettyMIDI(self.midi_dir + self.dense_midis[index])
        
        midi_seqs = pretty_midi_to_seq_chunks(midi)

        empty_section_idxs = np.where(midi_seqs[1,:] == 0)[0]
        M_db_clean = np.delete(M_db, empty_section_idxs, axis=1)
        midi_seqs_clean = np.delete(midi_seqs, empty_section_idxs, axis=1)

        if midi_seqs_clean.shape[1] == 0 or midi_seqs_clean.shape[1] != M_db_clean.shape[1]:
            return None
          
        return torch.tensor(M_db_clean), torch.tensor(midi_seqs_clean) #torch.tensor(M_db_clean[:,[chosen_chunk_idx],:]), torch.tensor(midi_seqs_clean[:, [chosen_chunk_idx]])

    def __getname__(self, index):
        return self.dense_midis[index]

    def __len__(self):
        return len(self.dense_midis)

# FIX COLLATE
def collate_fn(data, batch_size=4, collate_shuffle=True): # I think this should still work
  # data is a list of 2d tensors; concatenate and shuffle all list items
  data = list(filter(lambda x: x is not None, data))
  specs = [item[0] for item in data]
  midis = [item[1] for item in data]

  full_spec_list = torch.cat(specs, 1) # concatenate all data along the first axis
  full_midi_list = torch.cat(midis, 1)

  if collate_shuffle == True:
      rand_idx = torch.randperm(full_spec_list.shape[1])
      print("DATA SIZE", full_spec_list.shape[1], full_midi_list.shape[1])
      full_spec_list=full_spec_list[:,rand_idx,:]
      full_midi_list=full_midi_list[:,rand_idx]

  if full_spec_list.shape[1] > MAX_BATCH:
      #print("TRIMMING BATCH")
      full_midi_list = full_midi_list[:, :MAX_BATCH]
      full_spec_list = full_spec_list[:, :MAX_BATCH, :]

  print("FINAL DATA SIZE", full_spec_list.shape[1], full_midi_list.shape[1]) 

  return full_spec_list, full_midi_list

if __name__ == '__main__':
    midi_dir = './small_matched_data/midi/'
    audio_dir = './small_matched_data/raw_audio/'
    midi_list = os.listdir(midi_dir)

    dataset = AudioMidiDataset(midi_list, audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    print(dataset.__len__()) # only 25 files rn 
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,shuffle=True)
    for x,y in dataloader:
      print(x.shape,"Targets",y.shape,"\n")
