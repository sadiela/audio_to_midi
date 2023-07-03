import numpy as np
#import torch
import librosa 
import os
import matplotlib.pyplot as plt

''' Spectrogram stuff '''
SEG_LENGTH = 65407
SAMPLE_RATE = 16000
HOP_WIDTH = 128
NUM_MEL_BINS = 512 # input depth
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_WIDTH

def num_samples(desired_length_in_seconds, sr): 
   # if you want 3 second chunks, how many samples do you need
   return 1/sr * desired_length_in_seconds

def compute_spectrogram(samples):
  overlap = 1 - (HOP_WIDTH / FFT_SIZE)
  return librosa.feature.melspectrogram(y=samples, sr=SAMPLE_RATE, 
        hop_length=HOP_WIDTH, 
        n_fft=FFT_SIZE, 
        n_mels=NUM_MEL_BINS, 
        fmin=MEL_LO_HZ, fmax=7600.0)

def split_audio(signal, segment_length=SEG_LENGTH, pad_end=True, axis=-1):
    """ Split audio into frames """
    #print(signal.shape)
    signal_length = signal.shape[0]
    num_frames = signal_length // segment_length
    rest_samples = signal_length % segment_length
    if rest_samples !=0 and pad_end: 
      new_sig = np.zeros((num_frames+1)*segment_length)
      new_sig[0:signal_length] = signal
      signal = new_sig
    frames = signal.reshape(int(len(signal)/segment_length),segment_length)
    return frames
  
def plot_spec(M_db):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=M_db, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_WIDTH, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

def plot_whole_wav(y,sr):
    M =  librosa.feature.melspectrogram(y=y, sr=sr, 
                  hop_length=HOP_WIDTH, 
                  n_fft=FFT_SIZE, 
                  n_mels=NUM_MEL_BINS, 
                  fmin=MEL_LO_HZ, fmax=7600.0)
    M_db = librosa.power_to_db(M, ref=np.max)
    plot_spec(M_db)

def plot_wav_chunk(y,sr, chunk_num=0): 
    y = split_audio(y,SEG_LENGTH)
    print("NEW SIG SHAPE:", y.shape)
    M =  librosa.feature.melspectrogram(y=y, sr=sr, 
            hop_length=HOP_WIDTH, 
            n_fft=FFT_SIZE, 
            n_mels=NUM_MEL_BINS, 
            fmin=MEL_LO_HZ, fmax=7600.0)
    mel_transposed = np.transpose(M, (2, 0, 1))
    M_db = librosa.power_to_db(mel_transposed, ref=np.max)
    plot_spec(M_db[:,chunk_num,:].T) # first chunk

if __name__ == '__main__':
    raw_audio_folder = './small_matched_data/raw_audio/'
    file_list = os.listdir(raw_audio_folder)
    for f in file_list: 
        if f[-3:] == 'wav':
            print(f)
            y, sr = librosa.load(raw_audio_folder+f, sr=SAMPLE_RATE)
            print(len(y), sr)
            print("LENGTH:", int(len(y)/sr)//60, ':', int(len(y)/sr)%60)
            plot_whole_wav(y,sr)
            input("...")
            plot_wav_chunk(y,sr)
            input("...")
    # after sanity check, these lengths look good. 
    # Now we can see how it translates for the spectrograms



