import os
import pretty_midi
import numpy as np
import sys
import matplotlib.pyplot as plt
from timeit import default_timer as timer


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
for i in range(131,259):
    event_dictionary[i] = 'NOTE_END:', str(i-131)

for i in range(259,760):
    event_dictionary[i] = round(0.01*(i-258),2)

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
    for i, chunk in enumerate(seq_chunks):
        base_time = SEG_LENGTH_SECS*i # whatever chunk we are at
        cur_time = base_time
        for event in chunk: 
            if event in range(2, 130): # NOTE EVENT!
                print(event_dictionary[i])
                note = pretty_midi.Note(velocity=100, pitch=event-3, start=cur_time, end=cur_time + 0.25)
                cur_inst.notes.append(note)
            elif event in range(130,631): # TIME SHIFT!
                cur_time = base_time + event_dictionary[event]
            elif event == 0:
                break # end of this chunk
    
    cur_midi.instruments.append(cur_inst)
    return cur_midi
    #if not os.path.exists(target_dir + 'seq_conversion1.mid'):
    #    cur_midi.write(str(target_dir + 'seq_conversion1.mid'))

def pretty_midi_to_seq_chunks_w_offsets(open_midi): 
    note_starts = [note.start for note in open_midi.instruments[0].notes]
    note_ends = [note.end for note in open_midi.instruments[0].notes]
    num_segs = int((note_ends[-1] // SEG_LENGTH_SECS)) + 1
    event_sequences = [[2] for _ in range(num_segs)] 

    for note in open_midi.instruments[0].notes:
        note_start_seg = int((note.start // SEG_LENGTH_SECS))
        note_end_seg = int((note.end // SEG_LENGTH_SECS))

    
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences).T
    array_seqs = array_seqs.astype('int8')
    return array_seqs

def pretty_midi_to_seq_chunks(open_midi): 
    note_starts = [note.start for note in open_midi.instruments[0].notes]
    note_ends = [note.end for note in open_midi.instruments[0].notes]
    num_segs = int((note_ends[-1] // SEG_LENGTH_SECS)) + 1
    event_sequences = [[2] for _ in range(num_segs)] # Start with BOS TOKEN
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
            if rounded_offset != 0.0:
                event_sequences[cur_seg].append(event_idxs[rounded_offset])
        event_sequences[cur_seg].append(event_idxs["NOTE:"+str(note.pitch)])
        #input("Continue...")
        previous_note_time = note.start
    #print("SEQUENCE:", event_sequences, [event_dictionary[i] for i in event_sequences[cur_seg]])
    for seq in event_sequences:
        seq.append(0) # APPEND EOS TOKENS
        #print(len(seq))
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences).T
    array_seqs = array_seqs.astype('int8')
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
    seq_dir = './small_matched_data/sequences/'
    seq_files = os.listdir(seq_dir)

    # start with just note onset and time events!

    # Test by converting back to MIDI
    start_time = timer()
    for file in mid_files:
        #print(file)
        #print(midi_directory + file)
        open_midi = pretty_midi.PrettyMIDI(midi_directory + file)
        seq_chunks = pretty_midi_to_seq_chunks(open_midi)
        #seq_chunks_to_pretty_midi(seq_chunks, target_dir)
        #midi_to_wav(target_dir + 'seq_conversion1.mid', target_dir + 'seq_conv1.wav')
        #input("Continue...")
    end_time = timer()

    print("CONVERT ON THE FLY RUNTIME:",end_time-start_time)

    start_time2 = timer()
    for file in seq_files:
        seq = np.load(seq_dir + file)
    end_time2 = timer()

    print("LOAD SEQ RUNTIME:",end_time2-start_time2)
