import os 
import sys 
from pathlib import Path

SOUNDFONT_PATH = "./GeneralUser_GS_v1.471.sf2"

def midi_to_wav(midi_path,wav_path):
    cmd = "fluidsynth -F " + wav_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
    ret_status = os.system(cmd)
    if ret_status != 0:
        sys.exit(ret_status)

def midis_to_wavs(midi_dir, wav_dir=None, midi_files=None): # can provide list, don't need to, same with wav_dir
    if wav_dir is None: 
        wav_dir = midi_dir
    if midi_files is None: 
        midi_files = [midi for midi in os.listdir(midi_dir) if midi[-3:]=="mid"]
    for midi in midi_files: 
        output_path = wav_dir + midi[:-3]+'wav'
        midi_path = midi_dir  + midi
        if not os.path.isfile(output_path):
            midi_to_wav(midi_path, output_path)

def get_free_filename(stub, dir, suffix='', date=False):
    # Create unique file/directory 
    counter = 0
    while True:
        if date:
            file_candidate = '{}/{}-{}-{}{}'.format(str(dir), stub, datetime.today().strftime('%Y-%m-%d'), counter, suffix)
        else: 
            file_candidate = '{}/{}-{}{}'.format(str(dir), stub, counter, suffix)
        if Path(file_candidate).exists():
            #print("file exists")
            counter += 1
        else:  # No match found
            print("Counter:", counter)
            if suffix=='.p':
                print("will create pickle file")
            elif suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate


#if __name__ == '__main__':
