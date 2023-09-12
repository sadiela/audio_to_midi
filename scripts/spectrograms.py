import numpy as np
#import torch
import librosa 
import os
import matplotlib.pyplot as plt
from timeit import default_timer as timer

''' Spectrogram stuff '''
SEG_LENGTH = 65407
SAMPLE_RATE = 16000
HOP_WIDTH = 128
NUM_MEL_BINS = 512 # input depth
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_WIDTH
LEARNABLE_EOS = np.random.rand(1,NUM_MEL_BINS) # FIX

'''def num_samples(desired_length_in_seconds, sr): 
   # if you want 3 second chunks, how many samples do you need
   return 1/sr * desired_length_in_seconds'''

def split_audio(signal, segment_length=SEG_LENGTH, pad_end=True, axis=-1, chunks=None):
    """ Split audio into frames """
    signal_length = signal.shape[0]
    num_frames = signal_length // segment_length 
    rest_samples = signal_length % segment_length
    if rest_samples !=0 and pad_end: 
      new_sig = np.zeros((num_frames+1)*segment_length)
      new_sig[0:signal_length] = signal
      signal = new_sig
    frames = signal.reshape(int(len(signal)/segment_length),segment_length)
    if chunks is not None and max(chunks)<frames.shape[0]: 
        frames = frames[chunks,:]
    return frames
  
def plot_spec(M_db):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(data=M_db, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_WIDTH, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

def plot_whole_wav(y,sr):
    M_db, _ = calc_mel_spec(y=y, sr=sr, rand=False)
    plot_spec(M_db)

def plot_wav_chunk(y,sr, chunk_num=0): 
    M_db, _ = calc_mel_spec(y=y, sr=sr)
    plot_spec(M_db[:,chunk_num,:].T) # first chunk

def calc_mel_spec(audio_file=None, y=None, sr=None, chunks=None):
    # can take either a file or y,sr
    if audio_file is not None: 
        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    y = split_audio(y,SEG_LENGTH, chunks=chunks)

    # convert to melspectrograms
    M =  librosa.feature.melspectrogram(y=y, sr=sr, 
              hop_length=HOP_WIDTH, n_fft=FFT_SIZE, 
              n_mels=NUM_MEL_BINS, fmin=MEL_LO_HZ, fmax=7600.0)
    M_transposed = np.transpose(M, (0,2,1)) # transpose to be BATCH_SIZE x SEQ_LEN x  EMBED_DIM
    eos_block = LEARNABLE_EOS * np.ones((M_transposed.shape[0], 1, NUM_MEL_BINS)) # append EOS TO THE END OF EACH SEQUENCE!
    M_transposed = np.append(M_transposed, np.atleast_3d(eos_block), axis=1)
    M_db = librosa.power_to_db(M_transposed, ref=np.max) # logscale magnitudes
    return M_db

if __name__ == '__main__':
    raw_audio_folder = './small_matched_data/raw_audio/'
    spectrogram_folder = './small_matched_data/spectrograms/'
    file_list = os.listdir(raw_audio_folder)
    spec_list = os.listdir(spectrogram_folder)
    '''for f in file_list: 
        if f[-3:] == 'wav':
            y, sr = librosa.load(raw_audio_folder+f, sr=SAMPLE_RATE)
            print(len(y), sr)
            mel_spec = calc_mel_spec(raw_audio_folder + f)

            np.save(spectrogram_folder + f[:-4], mel_spec)'''
    
    for f in file_list: 
        if f[-3:] == 'wav':
            print(f)
            y, sr = librosa.load(raw_audio_folder+f, sr=SAMPLE_RATE)
            print(y.shape)
            y_chunks, idxs = split_audio(y, segment_length=SEG_LENGTH, max_frames=16)
            print(y_chunks.shape, idxs)
            input("Continue...")

    # TIME COMPARISON: 
    start_time = timer()
    for f in file_list: 
        if f[-3:] == 'wav':
            y, sr = librosa.load(raw_audio_folder+f, sr=SAMPLE_RATE)
            mel_spec = calc_mel_spec(raw_audio_folder + f)
    end_time = timer()
    print("converting on the fly:", end_time-start_time) 

    start2 = timer()
    for s in spec_list:
        seq = np.load(spectrogram_folder + s)
    end2 = timer()
    print("loading spec:", end2-start2)
    # after sanity check, these lengths look good. 
    # Now we can see how it translates for the spectrograms



