import os
import pretty_midi
import numpy as np

SEG_LENGTH_SECS = 4.0879375
BIN_QUANTIZATION = 0.01 # should only need 500 time events
SOUNDFONT_PATH = "./GeneralUser_GS_v1.471.sf2"
MAX_LENGTH=1024


event_dictionary = {}
event_dictionary[0] = '<EOS>'
event_dictionary[1] = '<PAD>'
event_dictionary[2] = '<BOS>'
for i in range(3,131):
    event_dictionary[i] = 'NOTE:' + str(i-3)

for i in range(131,632):
    event_dictionary[i] = round(0.01*(i-130),2)

event_idxs = {v: k for k, v in event_dictionary.items()}

def seq_chunks_to_pretty_midi(seq_chunks, target_dir):
    cur_midi = pretty_midi.PrettyMIDI() # define new midi object WITH PROPER TEMPO!!!
    cur_inst = pretty_midi.Instrument(program=1)
    for i, chunk in enumerate(seq_chunks):
        base_time = SEG_LENGTH_SECS*i # whatever chunk we are at
        cur_time = base_time
        for event in chunk: 
            if event in range(2, 130): # NOTE EVENT!
                note = pretty_midi.Note(velocity=100, pitch=event-2, start=cur_time, end=cur_time + 0.25)
                cur_inst.notes.append(note)
            elif event in range(130,631): # TIME SHIFT!
                cur_time = base_time + event_dictionary[event]
            elif event == 0:
                break # end of this chunk
    
    cur_midi.instruments.append(cur_inst)
    return cur_midi
    #if not os.path.exists(target_dir + 'seq_conversion1.mid'):
    #    cur_midi.write(str(target_dir + 'seq_conversion1.mid'))

def pretty_midi_to_seq_chunks(open_midi): 
    note_starts = [note.start for note in open_midi.instruments[0].notes]
    note_ends = [note.end for note in open_midi.instruments[0].notes]
    num_segs = int((note_ends[-1] // SEG_LENGTH_SECS)) + 1
    event_sequences = [[] for _ in range(num_segs)]
    cur_seg = 0
    previous_note_time = 0.0
    for note in open_midi.instruments[0].notes:
        if note.start > SEG_LENGTH_SECS*(cur_seg+1):
            #print("STARTING NEW SEGMENT")
            cur_seg = int((note.start // SEG_LENGTH_SECS))
        if note.start > previous_note_time and note.start > cur_seg*SEG_LENGTH_SECS: 
            # add a time event
            note_offset = note.start - cur_seg*SEG_LENGTH_SECS
            rounded_offset = round(note_offset, 2)
            #print("OFFSET AND ROUNDED:", note_offset, rounded_offset)
            #print("DICT EVENT:", event_idxs[rounded_offset])
            if rounded_offset != 0.0:
                event_sequences[cur_seg].append(event_idxs[rounded_offset])
        event_sequences[cur_seg].append(event_idxs["NOTE:"+str(note.pitch)])
        #input("Continue...")
        previous_note_time = note.start
    #print("SEQUENCE:", event_sequences, [event_dictionary[i] for i in event_sequences[cur_seg]])
    for seq in event_sequences:
        seq.append(0) # APPEND EOS TOKENS
        #print(len(seq))
    # NOW PAD THEM ALL TO THE LENGTH OF THE LONGEST ONE!
    longest_seq = max([len(seq) for seq in event_sequences])
    #print("LONGEST:", longest_seq)
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences).T
    return array_seqs

def midi_to_wav(midi_path,wav_path):
    print("CONVERTING")
    # using the default sound font in 44100 Hz sample rate
    #cmd = "fluidsynth -F " + wav_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
    cmd = "fluidsynth -F " + wav_path + ' ' + SOUNDFONT_PATH + ' ' + midi_path + ' -r 16000 -i'
    print(cmd)
    ret_status = os.system(cmd)


if __name__ == '__main__':
    midi_directory = './small_matched_data/midi/'
    mid_files = os.listdir(midi_directory)
    target_dir = './small_matched_data/midi_reconverted/'

    # start with just note onset and time events!

    # Test by converting back to MIDI
    for file in mid_files:
        print(file)
        print(midi_directory + file)
        open_midi = pretty_midi.PrettyMIDI(midi_directory + file)
        seq_chunks = pretty_midi_to_seq_chunks(open_midi)
        print("ARRAY SHAPE:", type(seq_chunks), seq_chunks.shape, seq_chunks[:,0], seq_chunks[0,:])
        #seq_chunks_to_pretty_midi(seq_chunks, target_dir)
        #midi_to_wav(target_dir + 'seq_conversion1.mid', target_dir + 'seq_conv1.wav')
        input("Continue...")
