import os
import pretty_midi
from tqdm import tqdm
#from numpy import ndarray
import sys
import multiprocessing

SOUNDFONT_PATH = "./GeneralUser_GS_v1.471.sf2"

# Python program to store list to file using pickle module
import pickle

# write list to binary file
def write_list(a_list, fname):
    # store list in binary file so 'wb' mode
    with open(fname, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

# Read list
def read_list(fname):
    # for reading also binary mode is important
    with open(fname, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def sep_and_crop(midi_directory, target_directory):
    # if folder doesn't exist, create it
    if not os.path.isdir(str(target_directory)):
        print("creating target directory")
        os.makedirs(str(target_directory))
# takes a directory filled with midi files, creates new midi files for each individual (NOT DRUM) track
    # in the original files, so each output midi has a single track
    file_list = os.listdir(midi_directory)
    num_tracks = 0 
    num_errors = 0
    print("NUMBER OF FILES:", len(file_list))
    for file in tqdm(file_list): 
        try:
            open_midi = pretty_midi.PrettyMIDI(str(midi_directory / file))
            bpm = open_midi.get_tempo_changes()[1][0]
            for i, instrument in enumerate(open_midi.instruments): 
                cur_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm) # define new midi object WITH PROPER TEMPO!!!
                cur_inst = pretty_midi.Instrument(program=1) # create new midi instrument
                # copy notes from instrument to cur 
                if not instrument.is_drum:
                    num_tracks += 1
                    start_time = instrument.notes[0].start
                    for note in instrument.notes:
                        shifted_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start-start_time, end=note.end-start_time)
                        cur_inst.notes.append(shifted_note)
                    # save cur as a new midi file
                    cur_midi.instruments.append(cur_inst)
                    if not os.path.exists(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')):
                        cur_midi.write(str(target_directory / str(file.split('.')[0] + '_'+ str(i) + '.mid')))
                    else:
                        print(file.split('.')[0] + '_'+ str(i) + '.mid EXISTS!')
        except Exception as e:
            #print("ERROR!", e)
            num_errors += 1
            pass
    print("NUMBER OF TRACKS:", num_tracks)
    print("NUMBER ERRORS:", num_errors)


'''def midi_to_wav(midi_path,wav_path):
        # using the default sound font in 44100 Hz sample rate
        fs = FluidSynth(sound_font=SOUNDFONT_PATH)
        fs.midi_to_audio(midi_path, wav_path)'''

# can parallelize these conversions since I have 8 CPU cores
def process_midis_to_wavs(midi_files, midi_dir, wav_dir):
    for midi in midi_files: 
        output_path = wav_dir + midi[:-3]+'wav'
        midi_path = midi_dir  + midi
        print("MIDI PATH AND OUTPUT PATH:", midi_path, output_path)
        input("Continue...")
        cmd = "fluidsynth -F " + output_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
        print(cmd)
        ret_status = os.system(cmd)
        if ret_status != 0:
            print("RETURN STATUS:", ret_status)
            sys.exit(ret_status)
        
    
def midis_to_wavs_multi(midi_list, midi_dir, wav_dir=None, num_processes=1): # 60!!!
    print("LIST LEN:", len(midi_list))
    # split into chunks:
    chunk_size = len(midi_list) // num_processes
    midi_chunks = [midi_list[i:i+chunk_size] for i in range(0, len(midi_list), chunk_size)]

    # create a list of processes
    print("Starting processes:", len(midi_chunks))
    print(midi_dir, wav_dir)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=process_midis_to_wavs, args=(midi_chunks[i],str(midi_dir), str(wav_dir)))
        process.start()
        processes.append(process)

    # wait for all processes to finish
    for process in processes:
        process.join()

def gen_small_dataset(midi_folder, raw_audio_folder): 
    midi_files = os.listdir(midi_folder)
    process_midis_to_wavs(midi_files, midi_folder, raw_audio_folder)

def convert_list(midi_stub, file_list, out_stub): 
    for t in file_list:
        midi_path = midi_stub + t 
        print(midi_path)
        output_path = out_stub + t[:-3] + 'wav'
        print(midi_path)
        if not os.path.isfile(output_path):
            cmd = "fluidsynth -F " + output_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
            ret_status = os.system(cmd)

if __name__ == '__main__':
    #small_midi = './small_matched_data/midi'
    #small_raw_audio = './small_matched_data/raw_audio'
    #gen_small_dataset(small_midi, small_raw_audio)
    #sys.exit(1)

    midi_list = read_list('./data_lists/dense_filtered.p')
    midi_dir = './lmd_tracks/'
    raw_audio_stub = '/raw_audio/'
    midis_to_wavs_multi(midi_list, midi_dir, wav_dir=raw_audio_stub, num_processes=48)
    print("DONE!")

    sys.exit(0)

    print("Converting tracks to raw audio")
    midi_stub = './lmd_full/'
    track_stub = './lmd_tracks/'
    raw_audio_stub = '/raw_audio/'

    folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'] # '0','1',

    # create directories
    #for f in folder_extensions:
    #    os.makedirs(raw_audio_stub + f, mode=0o777, exist_ok=True)

    # load 
    densefiles = read_list('./data_lists/densefiles.p')
    convert_list(track_stub, densefiles, raw_audio_stub)
    sys.exit(0)

    med_testfiles = read_list('./data_lists/testfiles_med.p')
    med_trainfiles = read_list('./data_lists/trainfiles_med.p')
    
    print("CONVERTING!")
    convert_list(track_stub, med_testfiles, raw_audio_stub)
    convert_list(track_stub, med_trainfiles, raw_audio_stub)

    

    sys.exit(0)

    e = '0'
    cur_track_folder = track_stub + e 
    cur_raw_audio = raw_audio_stub + e

    midis_to_wavs_multi(cur_track_folder, wav_dir=cur_raw_audio, num_processes=4)

    # separate MIDIs into tracks and crop empty starts
    '''for e in folder_extensions: 
        print("FOLDER", e)
        cur_midi_folder = midi_stub + e 
        cur_track_folder = track_stub + e 
        sep_and_crop(Path(cur_midi_folder), Path(cur_track_folder))'''

    # convert MIDIs to wavs
    #for e in folder_extensions:
    
    # get list of files in cur_raw_audio 
    #rawaudio_file_list = os.listdir(cur_raw_audio)
    #print(len(rawaudio_file_list))

    #write_list(rawaudio_file_list, 'converted_midis_new.p') # 11246 files
    #flist = read_list('converted_midis_new.p')
    #print('List is', len(flist), flist[0:25])


    '''small_midi = './small_matched_data/midi'
    small_raw_audio = './small_matched_data/raw_audio'

    raw_audio_files = ['26709364ebef444f80ce611a35cb04e0_5.wav', '21d50e25cc605b51c4a18addc1fa7a32_8.wav', '226fd106ad791a971d1d3c2a36164567_3.wav', '2146be0df514d22b66a801d164cc4392_14.wav', '23ae70e204549444ec91c9ee77c3523a_6.wav', '212abd583fe23890d0eae5f2b1f76aa2_0.wav', '236b22aca291dd04cbe49805e5104e09_0.wav', '20f7ffe70fc4cd23452603822318dae3_3.wav', '21e13737532a3730af28517d9836612e_5.wav', '24f40f00caa71bfcd0ab0fc3363b57c0_4.wav', '21276e9ddc992361a8b35edd5f118683_2.wav', '2577f484437ed2e01e053b6e8850bb7f_16.wav', '23660fbd19b6c5860d31eef6f1e2d1e1_1.wav', '20e763d48993f42f24d2daf6cf27068c_3.wav', '2403be827ea8072efd350d4dc4b9b0e3_5.wav', '205b4e193c2b1f2e7d63457fa61f5d1d_2.wav', '2470613bcf7a32c5874370245190653e_1.wav', '25d18450c9a3e5c4b6142362d67a2e3d_3.wav', '21b14d818ef5e347ff051e56f71f653a_12.wav', '2403fded92adb988145aac237dc5f6ee_3.wav', '25f924b83c209ebada9e4653bb44a714_5.wav', '2673f319aeb609cdad8b6ce30c39cf66_2.wav', '268957bbebbe84b3a8033f4e60957010_1.wav', '2147ff3314711da1cf12071335f7bfdc_1.wav', '20428139d7f12c0678e31fa27b42b342_5.wav']
    for f in raw_audio_files:
        print(f)
        shutil.copy(cur_track_folder + '/' + f[:-3] + 'mid', small_midi)
        shutil.copy(cur_raw_audio + '/' + f, small_raw_audio)

        # copy files into small_matched_data


    #midis_to_wavs_multi(cur_track_folder, cur_raw_audio, num_processes=4)
    #print("FINSIHED", e)'''

