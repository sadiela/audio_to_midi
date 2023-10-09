import os
import pretty_midi
import numpy as np
import sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from operator import itemgetter
import pickle
from tqdm import tqdm
import multiprocessing
from utility import *


SEG_LENGTH_SECS = 4.0879375
BIN_QUANTIZATION = 0.01 # should only need 500 time events
MAX_LENGTH=1024

event_dictionary = {}
event_dictionary[0] = '<EOS>'
event_dictionary[1] = '<PAD>'
event_dictionary[2] = '<BOS>'
for i in range(3,131):
    event_dictionary[i] = 'NOTE:' + str(i-3)

for i in range(131,632):
    event_dictionary[i] = round(0.01*(i-130),2)

#event_idxs = {v: k for k, v in event_dictionary.items()}
#for i in range(259,760):
#    event_dictionary[i] = round(0.01*(i-258),2)

event_idxs = {v: k for k, v in event_dictionary.items()}

def custom_plot_pianoroll(
    midi_object, 
    minc: int = -2,maxc: int = 7,
    resolution: int = 24,
    cmap: str = "Blues",
    grid_axis: str = "both",
    grid_linestyle: str = ":",
    grid_linewidth: float = 0.5,
    vmax=1,ymin=None,ymax=None,
    **kwargs,
    ):

    pianoroll = midi_object.instruments[0].get_piano_roll()
    _, ax = plt.subplots()


    img = ax.imshow(pianoroll,
        cmap=cmap,aspect="auto",vmin=0,
        vmax=vmax, # if pianoroll.dtype == np.bool_ else 127,
        origin="lower",interpolation="none",
        **kwargs,
    )

    ax.set_yticks(np.arange(12*(minc+2), 12*(maxc+3), 12))
    ax.set_yticklabels([f"C{minc+i}" for i in range(maxc-minc+1)], fontsize=12)

    nonzero_row_indices = np.nonzero(np.count_nonzero(pianoroll, axis=1))
    if not ymin:
        ymin = np.min(nonzero_row_indices) - 12
    if not ymax: 
        ymax = np.max(nonzero_row_indices) + 12

    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Pitch", fontsize=14)

    # Format x-axis
    ax.set_xticks(np.arange(-0.5, pianoroll.shape[1], resolution)) # put labels
    ax.set_xticklabels(np.arange(0, pianoroll.shape[1]//resolution +1, 1), fontsize=12)
    ax.set_xlim([-0.5, pianoroll.shape[1]])
    ax.set_xlabel("Time (beats)", fontsize=14)

    if grid_axis != "off":
        ax.grid(
            axis='x', # or "both"
            color="k",
            linestyle=grid_linestyle,
            linewidth=grid_linewidth,
        )

    plt.show()
    return img

def seq_chunks_to_pretty_midi(seq_chunks):
    cur_midi = pretty_midi.PrettyMIDI() # define new midi object WITH PROPER TEMPO!!!
    cur_inst = pretty_midi.Instrument(program=1)
    print(len(seq_chunks), seq_chunks[0])
    #seq_chunks = seq_chunks.T
    for i, chunk in enumerate(seq_chunks):
        base_time = SEG_LENGTH_SECS*i # whatever chunk we are at
        cur_time = base_time
        for event in chunk: 
            if event in range(3, 131): # NOTE EVENT!
                note = pretty_midi.Note(velocity=100, pitch=event-3, start=cur_time, end=cur_time + 0.25)
                cur_inst.notes.append(note)
            elif event in range(131,632): # TIME SHIFT!
                cur_time = base_time + event_dictionary[event]
            elif event == 0:
                break # end of this chunk
    
    cur_midi.instruments.append(cur_inst)
    return cur_midi
    #if not os.path.exists(target_dir + 'seq_conversion1.mid'):
    #    cur_midi.write(str(target_dir + 'seq_conversion1.mid'))

def seq_chunks_w_noteoff_to_prettymidi(seq_chunks):
    cur_midi = pretty_midi.PrettyMIDI() # define new midi object WITH PROPER TEMPO!!!
    cur_inst = pretty_midi.Instrument(program=1)
    unended_notes = []
    print(seq_chunks.shape)
    seq_chunks = seq_chunks.T
    for i, chunk in enumerate(seq_chunks):
        base_time = SEG_LENGTH_SECS*i # whatever chunk we are at
        cur_time = base_time
        for event in chunk: 
            if event in range(3, 130): # NOTE EVENT!
                # Check if note is in unended note list
                unended_note = None
                for i,note in enumerate(unended_notes):
                    if event-3 == note["pitch"]:
                        unended_note = unended_notes.pop(i) # removes it from the unended note list
                        break
                if unended_note is not None:
                    # END THE NOTE AND ADD IT TO THE DICT
                    note = pretty_midi.Note(velocity=100, pitch=event-3, start=unended_note['start'], end=cur_time)
                    cur_inst.notes.append(note)
                else: # NOTE START
                    new_note = {}
                    new_note["pitch"] = event-3
                    new_note["start"] = cur_time 
                    unended_notes.append(new_note)
            elif event in range(130,631): # TIME SHIFT!
                cur_time = base_time + event_dictionary[event]
            elif event == 0:
                break # end of this chunk
    cur_midi.instruments.append(cur_inst)
    return cur_midi

def pretty_midi_to_seq_chunks_w_noteoff(open_midi): 
    note_starts = [(note.pitch,note.start, "START") for note in open_midi.instruments[0].notes]
    note_ends = [(note.pitch,note.end, "END") for note in open_midi.instruments[0].notes]
    all_note_tuples = note_starts + note_ends
    all_note_tuples = sorted(all_note_tuples,key=itemgetter(1))
    num_segs = int((open_midi.get_end_time() // SEG_LENGTH_SECS)) + 1
    event_sequences = [[2] for _ in range(num_segs)] 
    previous_note_time = 0.0
    for note in all_note_tuples:
        cur_seg = int((note[1] // SEG_LENGTH_SECS))
        if note[1] > previous_note_time:
            note_offset = note[1] - cur_seg*SEG_LENGTH_SECS
            rounded_offset = round(note_offset, 2)
            if rounded_offset != 0.0:
                event_sequences[cur_seg].append(event_idxs[rounded_offset])
        event_sequences[cur_seg].append(event_idxs["NOTE:"+str(note[0])])
        previous_note_time = note[1]
    
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences)
    #array_seqs = array_seqs.astype('int8')
    #print(list(array_seqs[:,0]))
    return array_seqs

def pretty_midi_to_seq_chunks(midi_obj): 
    num_segs = int((midi_obj.get_end_time() // SEG_LENGTH_SECS)) + 1
    event_sequences = [[2] for _ in range(num_segs)] # Start with BOS TOKEN
    cur_seg = 0
    previous_note_time = 0.0
    for note in midi_obj.instruments[0].notes:
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
        previous_note_time = note.start
    #print("SEQUENCE:", event_sequences, [event_dictionary[i] for i in event_sequences[cur_seg]])
    for seq in event_sequences:
        seq.append(0) # APPEND EOS TOKENS
        #print(len(seq))
    # NOW PAD THEM ALL TO THE LENGTH OF THE LONGEST ONE!
    #longest_seq = max([len(seq) for seq in event_sequences])
    #print("LONGEST:", longest_seq)
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences)
    array_seqs = array_seqs.astype('int16')
    return array_seqs


def convert_midis(seq_stub, midi_list, midi_stub): 
    for mid in midi_list:
        if not os.path.exists(seq_stub + mid[:-3] + 'npy'):
            open_midi = pretty_midi.PrettyMIDI(midi_stub + mid)
            seq_chunks = pretty_midi_to_seq_chunks(open_midi)
            np.save(seq_stub + mid[:-3] + 'npy', seq_chunks)

def midis_to_seqs_multi(seq_stub, midi_list, midi_stub, num_processes=60): # 60!!!
    print("LIST LEN:", len(midi_list))
    # split into chunks:
    chunk_size = len(midi_list) // num_processes
    midi_chunks = [midi_list[i:i+chunk_size] for i in range(0, len(midi_list), chunk_size)]

    # create a list of processes
    print("Starting processes:", len(midi_chunks))
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=convert_midis, args=(seq_stub,midi_chunks[i], midi_stub))
        process.start()
        processes.append(process)

    # wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == '__main__':

    print("Converting MIDIs to seqs")
    midi_stub = './lmd_tracks/'
    seq_stub = '/sequences/'

    folder_extensions = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'] # '0','1',

    # create directories
    for f in folder_extensions:
        os.makedirs(seq_stub + f, mode=0o777, exist_ok=True)

    with open('./data_lists/dense_filtered.p', 'rb') as fp:
        dense_midis = pickle.load(fp)

    print("NUM MIDS:", len(dense_midis))
    midis_to_seqs_multi(seq_stub, dense_midis, midi_stub, num_processes=1)

    sys.exit(0)

    midi_directory = './small_matched_data/midi/'
    mid_files = os.listdir(midi_directory)
    target_dir = './small_matched_data/midi_reconverted/'
    noteoff_dir = './small_matched_data/midi_reconverted/'
    seq_dir = './small_matched_data/sequences/'
    seq_files = os.listdir(seq_dir)

    # start with just note onset and time events!

    # Test by converting back to MIDI
    #start_time = timer()
    for file in mid_files:
        print(file)
        #print(midi_directory + file)
        open_midi = pretty_midi.PrettyMIDI(midi_directory + file)
        seq_chunks = pretty_midi_to_seq_chunks(open_midi)
        new_midi = seq_chunks_to_pretty_midi(seq_chunks)
        # Save midi
        new_midi.write(noteoff_dir + file)
        #seq_chunks_to_pretty_midi(seq_chunks, target_dir)
        #midi_to_wav(target_dir + 'seq_conversion1.mid', target_dir + 'seq_conv1.wav')
    #end_time = timer()

    sys.exit(0)

    print("CONVERT ON THE FLY RUNTIME:",end_time-start_time)

    start_time2 = timer()
    for file in seq_files:
        seq = np.load(seq_dir + file)
    end_time2 = timer()

    print("LOAD SEQ RUNTIME:",end_time2-start_time2)
