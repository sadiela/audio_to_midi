import os 
import sys 

SOUNDFONT_PATH = "./GeneralUser_GS_v1.471.sf2"

def midi_to_wav(midi_path,wav_path):
    cmd = "fluidsynth -F " + wav_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
    ret_status = os.system(cmd)
    if ret_status != 0:
        sys.exit(ret_status)

def midis_to_wavs(midi_dir, wav_dir=None, midi_files=None): # can provide list, don't need to, same with wav_dir
    if wav_path is None: 
        wav_path = midi_path
    if midi_files is None: 
        midi_files = [midi for midi in os.listdir(midi_dir) if midi[-3:]=="mid"]
    for midi in midi_files: 
        output_path = wav_dir + midi[:-3]+'wav'
        midi_path = midi_dir  + midi
        if not os.path.isfile(output_path):
            midi_to_wav(midi_path, output_path)


if __name__ == '__main__':
    print("main")