import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle 
import os
import numpy as np

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
        # choose random file path from directory (not already chosen), chunk it 
        #cur_data = torch.load(self.paths[index])
        #print(self.paths[index])
        with open(self.paths[index], 'rb') as f:
          pickled_tensor = pickle.load(f)
        cur_data = torch.tensor(pickled_tensor.toarray()).clone().detach().float()

        p, l_i = cur_data.shape
        # make sure divisible by l
        # CHUNK! 
        #print("DATA SHAPE:", cur_data.shape)
        if l_i // self.l == 0: 
          padded_data = torch.zeros((p, self.l))
          padded_data[:,0:l_i] = cur_data
          l_i=self.l
        else: 
          padded_data = cur_data[:,:(cur_data.shape[1]-(cur_data.shape[1]%self.l))]
        
        chunked = torch.reshape(padded_data, (l_i//self.l,1, p, self.l)) 
        chunked = chunked[chunked.sum(dim=(2,3)) != 0]         # Remove empty areas
        chunked = torch.reshape(chunked, (chunked.shape[0], 1, p, self.l)) 

        if chunked.shape[0] != 0:
            return chunked # 3d tensor: l_i\\l x p x l
        else:
            return None

    def __getname__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

def collate_fn(data, collate_shuffle=True): # I think this should still work
  # data is a list of 2d tensors; concatenate and shuffle all list items
  data = list(filter(lambda x: x is not None, data))
  #print(len(data))
  full_list = torch.cat(data, 0) # concatenate all data along the first axis
  if collate_shuffle:
    idx = torch.randperm(full_list.shape[0])
    return  full_list[idx].view(full_list.size())
  else:
    return full_list