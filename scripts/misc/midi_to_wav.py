import os
import sys
import multiprocessing
import pickle

SOUNDFONT_PATH = "./GeneralUser_GS_v1.471.sf2"

def midi_to_wav(midi_path,wav_path):
    cmd = "fluidsynth -F " + wav_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
    ret_status = os.system(cmd)
    if ret_status != 0:
        sys.exit(ret_status)

def midis_to_wavs(midi_dir, wav_dir, midi_files):
    for midi in midi_files: 
        output_path = wav_dir + midi[:-3]+'wav'
        midi_path = midi_dir  + midi
        if not os.path.isfile(output_path):
            midi_to_wav(midi_path, output_path)

def chunk_list(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]
        
def midis_to_wavs_multi(midi_list, midi_dir, wav_dir=None, num_processes=1): 
    # split into chunks:
    midi_chunks = list(chunk_list(midi_list, num_processes))

    # create a list of processes
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=midis_to_wavs, args=(str(midi_dir), str(wav_dir),midi_chunks[i]))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

if __name__ == '__main__':
    data_list_path = '/Users/sadiela/Documents/phd/research/music/audio_to_midi/data_lists/chunked_datalists/trainfiles_0.p'
    midi_dir = '/Users/sadiela/Documents/phd/research/music/audio_to_midi/lmd_tracks/'
    raw_audio_dir = '/Users/sadiela/Documents/phd/research/music/audio_to_midi/scc_audio/'

    with open(data_list_path, 'rb') as fp:
        data_list = pickle.load(fp)

    # create subdirectories for raw audio directory (RUN ONCE THEN YOU CAN COMMENT THIS OUT)
    folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'] 
    for f in folder_extensions:
        os.makedirs(raw_audio_dir + f, mode=0o777, exist_ok=True)

    # you can use more than 4 processes depending on how many cores your computer has
    midis_to_wavs_multi(data_list,midi_dir=midi_dir, wav_dir=raw_audio_dir, num_processes=4)

    # These should match... for whatever subdir you are looking at
    print("LIST LEN:", len(data_list), "CONVERTED FILES:", len(os.listdir(raw_audio_dir+'X')))
