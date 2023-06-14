from ddsp import spectral_ops
import torch

''' Spectrogram stuff '''
SAMPLE_RATE = 16000
HOP_WIDTH = 128
NUM_MEL_BINS = 512 # input depth
FFT_SIZE = 2048
MEL_LO_HZ = 20.0
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_WIDTH

def compute_spectrogram(samples):
  overlap = 1 - (HOP_WIDTH / FFT_SIZE)
  return spectral_ops.compute_logmel(
      samples,
      bins=NUM_MEL_BINS,
      lo_hz=MEL_LO_HZ,
      overlap=overlap,
      fft_size=FFT_SIZE,
      sample_rate=SAMPLE_RATE)

def split_audio(signal, pad_end=False, pad_value=0, axis=-1):
    """ Split audio into frames """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = HOP_WIDTH - HOP_WIDTH
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(HOP_WIDTH - frames_overlap)
        pad_size = int(HOP_WIDTH - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames=signal.unfold(axis, HOP_WIDTH, HOP_WIDTH)
    return frames
  
 def flatten_frames(frames):
    return torch.reshape(frames, (-1,))
  
