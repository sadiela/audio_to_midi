# MIDI Transcription with Transformers

This project is an attempt to improve on the results obtained by Hawthorne et al. in "Sequence-to-Sequence Piano Transcription with Transformers" [1]. 

We plan to improve the model by pretraining it on a large synthetic dataset of paired raw audio and midi files (using a variety of synthetic instrument sounds) and fine-tuning it later for more specific tasks (e.g., piano transcription with the MAESTRO dataset). 

## To Do: 
- [x]  Design MIDI to raw audio pipeline using FluidSynth
- [x]  Implement T5-like model in pytorch
- [x]  Figure out I/O (spectrograms and MIDI-event output)

## Synthetic Data Generation: MIDI to Raw Audio with FluidSynth
To generate a large dataset of aligned MIDI and raw audio, we can synthesize MIDI from the Lakh MIDI Dataset [2] using FluidSynth.

## Implement T5-esque Transformer in Pytorch

## I/O
**Input**: spectrogram frames, one frame per input position
* Audio sample rate: 16kHz, FFT length 2048 samples, hop width 128 samples
* Scaled output to 512 mel bins to match models embedding size, used log-scaled magnitude
* Input sequences limited to 512 positions (511 frames plus EOS)
* Terminate input sequences with learnable EOS embedding
**Output**: softmax distribution over discrete vocabulary of event
* Heavily inspired by the messages defined in the MIDI specification
* **Vocabulary**:
    * Note: [128 values] indicates note-on or note-off event for one of 128 MIDI pitches
    * Velocity: [128 values] indicates a velocity change to be applied to all subsequent note events until the next velocity event
    * Time: [6000 values] absolute time location w/in a segment, quantized into 10ms bins; this time will apply to all subsequent note events until the next time event
         * Must appear in chronological order
         * Define vocab with times up to 60 seconds for flexibility, but because time resets for each segment, in practice we use only the first hundred events of this type
    * EOS: [1 value] Indicates the end of the sequence

## Parameters/Architecture Details from Hawthorne et al. 
- Embedded size: 512
- Feed-forward output dim: 1024
- Key/value dimensionality: 64
- 6 headed attention
- 8 layers each in the encoder and decoder
- Used float32 activations for better training stability


[1] @article{hawthorne2021sequence,
  title={Sequence-to-sequence piano transcription with transformers},
  author={Hawthorne, Curtis and Simon, Ian and Swavely, Rigel and Manilow, Ethan and Engel, Jesse},
  journal={arXiv preprint arXiv:2107.09142},
  year={2021}}

[2] @book{lakh,
  title={Learning-based methods for comparing sequences, with applications to audio-to-midi alignment and matching},
  author={Raffel, Colin},
  year={2016},
  publisher={Columbia University}}

[3] @inproceedings{
  maestrodataset,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},}
