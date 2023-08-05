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
    def __init__(self, dense_midis, audio_file_dir='/raw_audio/', midi_file_dir='./lmd_tracks/', rand=True):
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
        self.rand=True

    def __getitem__(self, index):
        # MELSPECTROGRAMS
        midi = pretty_midi.PrettyMIDI(self.midi_file_dir + dense_midis[index])
        midi_seqs = pretty_midi_to_seq_chunks(midi)
        idxs = np.where(midi_seqs[:,1] != 0)[0] # 0 is EOS TOKEN!, looks at 2nd row, checks for EOS's, thats empty sections
        if self.rand:
            idxs = np.random.permutation(idxs)
        if len(idxs) > MAX_BATCH:
            idxs = idxs[:MAX_BATCH]
        midi_seqs = midi_seqs[idxs,:]

        if midi_seqs.shape[1] == 0: #or midi_seqs_clean.shape[1] != M_db_clean.shape[1]:
            return None
        
        M_db = calc_mel_spec(audio_file = self.audio_dir + self.dense_midis[index][:-3] + 'wav', chunks=idxs)
        
          
        return torch.tensor(M_db), torch.tensor(midi_seqs) #torch.tensor(M_db_clean[:,[chosen_chunk_idx],:]), torch.tensor(midi_seqs_clean[:, [chosen_chunk_idx]])

    def __getname__(self, index):
        return self.dense_midis[index]

    def __len__(self):
        return len(self.dense_midis)

# FIX COLLATE
# probably won't need this anymore
def collate_fn(data, batch_size=1, collate_shuffle=True): # I think this should still work
  # data is a list of 2d tensors; concatenate and shuffle all list items
  data = list(filter(lambda x: x is not None, data))
  specs = [item[0] for item in data]
  midis = [item[1] for item in data]

  full_spec_list = torch.cat(specs, 1) # concatenate all data along the first axis
  full_midi_list = torch.cat(midis, 1)

  if collate_shuffle == True:
      rand_idx = torch.randperm(full_spec_list.shape[1])
      #print("DATA SIZE", full_spec_list.shape[1], full_midi_list.shape[1])
      full_spec_list=full_spec_list[:,rand_idx,:]
      full_midi_list=full_midi_list[:,rand_idx]

  if full_spec_list.shape[1] > MAX_BATCH:
      #print("TRIMMING BATCH")
      full_midi_list = full_midi_list[:, :MAX_BATCH]
      full_spec_list = full_spec_list[:, :MAX_BATCH, :]

  #print("FINAL DATA SIZE", full_spec_list.shape[1], full_midi_list.shape[1]) 

  return full_spec_list, full_midi_list

if __name__ == '__main__':
    midi_dir = './small_matched_data/midi/'
    audio_dir = './small_matched_data/raw_audio/'
    midi_list = os.listdir(midi_dir)

    for mid in midi_list: 
        midi = pretty_midi.PrettyMIDI(midi_dir + mid)
        midi_seqs = pretty_midi_to_seq_chunks(midi)
        print("MIDI SEQ SHAPE:", midi_seqs.shape)
        input("Continue...")
        idxs = np.where(midi_seqs[:,1] != 0)[0] # 0 is EOS TOKEN!, looks at 2nd row, checks for EOS's, thats empty sections
        print("IDXS", idxs)
        idxs = np.random.permutation(idxs)
        print("IDXS", idxs)
        if len(idxs) > MAX_BATCH:
            idxs = idxs[:MAX_BATCH]
        print("IDXS", idxs)
        midi_seqs = midi_seqs[idxs,:]
        
        M_db = calc_mel_spec(audio_file = audio_dir + mid[:-3] + 'wav', chunks=idxs)

        print(midi_seqs.shape, M_db.shape)
        input("Continue...")
        
        

    dataset = AudioMidiDataset(midi_list, audio_file_dir=audio_dir, midi_file_dir=midi_dir)
    print(dataset.__len__()) # only 25 files rn 
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,shuffle=True)
    for x,y in dataloader:
      print(x.shape,"Targets",y.shape,"\n")
