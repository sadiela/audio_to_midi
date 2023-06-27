import numpy as np
import torch
import librosa 
import os

''' Spectrogram stuff '''
SAMPLE_RATE = 16000
HOP_WIDTH = 128
NUM_MEL_BINS = 512 # input depth
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_WIDTH

def compute_spectrogram(samples):
  overlap = 1 - (HOP_WIDTH / FFT_SIZE)
  return librosa.feature.melspectrogram(y=samples, sr=SAMPLE_RATE, 
        hop_length=HOP_WIDTH, 
        n_fft=FFT_SIZE, 
        n_mels=NUM_MEL_BINS, 
        fmin=MEL_LO_HZ, fmax=7600.0)

def split_audio(signal, hop_width=128, pad_end=True, pad_value=0, axis=-1):
    """ Split audio into frames """
    print(signal.shape)
    signal_length = signal.shape[0]
    num_frames = signal_length // hop_width
    rest_samples = signal_length % hop_width
    if rest_samples !=0 and pad_end: 
      new_sig = np.zeros((num_frames+1)*hop_width)
      new_sig[0:signal_length] = signal
      signal = new_sig
    frames = signal.reshape(int(len(signal)/hop_width),hop_width)
    return frames
  
def flatten_frames(frames):
    return torch.reshape(frames, (-1,))
  
if __name__ == '__main__':
    raw_audio_folder = './small_matched_data/raw_audio/'
    file_list = os.listdir(raw_audio_folder)
    for f in file_list: 
        y, sr = librosa.load(raw_audio_folder+f, sr=SAMPLE_RATE)
        print(len(y), type(y), sr)


