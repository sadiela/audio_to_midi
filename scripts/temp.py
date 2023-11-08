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
import evaluation
import math
import mir_eval
np.set_printoptions(threshold=sys.maxsize)


SEG_LENGTH_SECS = 4.0879375
BIN_QUANTIZATION = 0.01 # should only need 500 time events
MAX_LENGTH=1024

event_dictionary = {}
event_dictionary[0] = '<EOS>'
event_dictionary[1] = '<PAD>'
event_dictionary[2] = '<BOS>'
for i in range(3,131):
    event_dictionary[i] = 'NOTE_ON:' + str(i-3)

for i in range(131,632):
    event_dictionary[i] = round(0.01*(i-130),2)

for i in range(632,760):
    event_dictionary[i] = 'NOTE_OFF:' + str(i-632)
    
for i in range(760,888):
    event_dictionary[i] = 'VEL:' + str(i-760)



#event_idxs = {v: k for k, v in event_dictionary.items()}
#for i in range(259,760):
#    event_dictionary[i] = round(0.01*(i-258),2)

event_idxs = {v: k for k, v in event_dictionary.items()}

def pretty_midi_to_seq_chunks_w_noteoff(open_midi): 
    note_starts = [(1,note.pitch,note.start) for note in open_midi.instruments[0].notes]
    note_ends = [(0,note.pitch,note.end) for note in open_midi.instruments[0].notes]
    all_note_tuples = note_starts + note_ends
    all_note_tuples = sorted(all_note_tuples,key=itemgetter(2))
    num_segs = int((open_midi.get_end_time() // SEG_LENGTH_SECS)) + 1
    event_sequences = [[2] for _ in range(num_segs)] 
    previous_note_time = 0.0
    # print(all_note_tuples)
    for note in all_note_tuples:
        cur_seg = int((note[2] // SEG_LENGTH_SECS))
        if note[2] > previous_note_time:
            note_offset = note[2] - cur_seg*SEG_LENGTH_SECS
            rounded_offset = round(note_offset, 2)
            if rounded_offset != 0.0:
                event_sequences[cur_seg].append(event_idxs[rounded_offset])
        if note[0] == 1:
            event_sequences[cur_seg].append(event_idxs["NOTE_ON:"+str(note[1])])
        elif note[0] == 0:
            event_sequences[cur_seg].append(event_idxs["NOTE_OFF:"+str(note[1])])
        previous_note_time = note[2]
    
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences)
    #array_seqs = array_seqs.astype('int8')
    #print(list(array_seqs[:,0]))
    return array_seqs

def pretty_midi_to_seq_chunks_w_noteoff_and_velocity(open_midi): 
    note_starts = [(1,note.pitch,note.start, note.velocity) for note in open_midi.instruments[0].notes]
    note_ends = [(0,note.pitch,note.end,note.velocity) for note in open_midi.instruments[0].notes]
    all_note_tuples = note_starts + note_ends
    all_note_tuples = sorted(all_note_tuples,key=itemgetter(2))
    num_segs = int((open_midi.get_end_time() // SEG_LENGTH_SECS)) + 1
    event_sequences = [[2] for _ in range(num_segs)] 
    previous_note_time = 0.0
    current_velocity = -1
    # print(all_note_tuples)
    for note in all_note_tuples:
        cur_seg = int((note[2] // SEG_LENGTH_SECS))
        if note[2] > previous_note_time:
            note_offset = note[2] - cur_seg*SEG_LENGTH_SECS
            
            note_offset *= 100
            rounded_offset = round(note_offset)/100.0
            # print(rounded_offset)
            if rounded_offset != 0.0:
                event_sequences[cur_seg].append(event_idxs[rounded_offset])
        if not current_velocity == note[3]:
            event_sequences[cur_seg].append(event_idxs["VEL:"+str(note[3])])
            current_velocity = note[3]
        if note[0] == 1:
            event_sequences[cur_seg].append(event_idxs["NOTE_ON:"+str(note[1])])
        elif note[0] == 0:
            event_sequences[cur_seg].append(event_idxs["NOTE_OFF:"+str(note[1])])
        previous_note_time = note[2]
    
    for seq in event_sequences:
        while (len(seq)) < MAX_LENGTH:
            seq.append(1) # PADDING!
    array_seqs = np.array(event_sequences)
    #array_seqs = array_seqs.astype('int8')
    #print(list(array_seqs[:,0]))
    return array_seqs

#array_seqs is an array of sequences
def seq_chunks_w_noteoff_and_velocity_to_pretty_midi(array_seqs):
    created_midi = pretty_midi.PrettyMIDI()
    # piano_program = pretty_midi.instrument_name_to_program('Piano 1')
    piano = pretty_midi.Instrument(program=1)
    currentNotes = []
    currentTime = 0.0
    currentVelocity = 100
    for i in range(len(array_seqs)):
        chunk = array_seqs[i]
        currentTime = SEG_LENGTH_SECS * i
        for event in chunk:
            if event >=3 and event < 131 :
                currentNotes.append((event, currentTime))
            elif event >= 632 and event < 760:
                for note_event in range(len(currentNotes)): 
                    if currentNotes[note_event][0] >=3 and currentNotes[note_event][0] <= 130 and currentNotes[note_event][0]-3 == event - 632:
                        if currentTime - currentNotes[note_event][1]  > 0:
                            note = pretty_midi.Note(velocity=currentVelocity, pitch = currentNotes[note_event][0]-3, start = currentNotes[note_event][1], end = currentTime)
                            piano.notes.append(note)
                        currentNotes.remove(currentNotes[note_event])
                        break
            elif event >= 131 and event < 632:
                currentTime = SEG_LENGTH_SECS*i + event_dictionary[event]
            elif event in range(760,888):
                currentVelocity = event-760
        
    created_midi.instruments.append(piano)
    return created_midi

def compare_two_prs(pr1, pr2): 
    pr1 = np.where(pr1 > 0, 1 ,0)
    pr2 = np.where(pr2 > 0, 1 ,0)
    minShape = min(pr2.shape[1],pr1.shape[1])
    return np.mean(np.square(pr1[:,:minShape]-pr2[:,:minShape]))

def testEval(refMid, estMid):
    ref_intervals,ref_pitches,ref_velocities = evaluation.get_intervals_and_pitches_and_velocities(refMid)
    est_intervals,est_pitches,est_velocities = evaluation.get_intervals_and_pitches_and_velocities(estMid)
    print(mir_eval.transcription_velocity.precision_recall_f1_overlap(ref_intervals,ref_pitches,ref_velocities,est_intervals,est_pitches,est_velocities))
    print(compare_two_prs(refMid.get_piano_roll(100,None,None), estMid.get_piano_roll(100,None,None)))
    custom_plot_pianoroll(refMid)
    custom_plot_pianoroll(estMid)

def genTestMidi():
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/evaluation_test_data/'
    filename = "polyphonicLong.mid"
    created_midi = pretty_midi.PrettyMIDI()
    # piano_program = pretty_midi.instrument_name_to_program('Piano 1')
    piano = pretty_midi.Instrument(program=1)
    for i in range(8):
        note = pretty_midi.Note(velocity=100, pitch = 72 + i, start = i, end = i+2)
        piano.notes.append(note)

    created_midi.instruments.append(piano)
    created_midi.write(gendirectory+filename)

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

# Simple monophonic midi with constant velocity < 4 seconds
# need to compare piano rolls visually
# l2 distance of generated and source piano rolls
# look at f1 score               
def test1():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'monophonic.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    testEval(tempMid1,asd)
    

# Polyphonic < 4 seconds constant velocity
def test2():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'polyphonic.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)

# Monophonic > 4 seconds constant velocity
def test3():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'monophonicLong.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)

# Polyphonic > 4 seconds constant velocity
def test4():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'polyphonicLong.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)

# Monophonic < 4 seconds varying velocity
def test5():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'monophonicShortVary.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)
    
def test6():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'polyphonicShortVary.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)

def test7():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'monophonicLongVary.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)

def test8():
    directory = os.path.dirname(os.path.abspath(__file__)) +'/evaluation_test_data/'
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    filename = 'polyphonicLongVary.mid'
    f = os.path.join(directory, filename)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    asd.write(gendirectory + filename)
    
    testEval(tempMid1,asd)



def runTests():
    print('Test1: ')
    test1()
    print('Test2: ')
    test2()
    print('Test3: ')
    test3()
    print('Test4: ')
    test4()
    print('Test5: ')
    test5()
    print('Test6: ')
    test6()
    print('Test7: ')
    test7()
    print('Test8: ')
    test8()

def runTestOnSubset():
    meanL2 = 0.0
    fileCounter = 0
    averageF1Score = 0
    directory = "/scratch2/lmd_tracks/lmd_tracks/d"
    for file in tqdm(os.listdir(directory)):
        if file.startswith("d00"):
            # print(file)
            f = os.path.join(directory, file)
            tempMid1 = pretty_midi.PrettyMIDI(f)
            thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
            asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
            meanL2 += compare_two_prs(tempMid1.get_piano_roll(100,None,None),asd.get_piano_roll(100,None,None))
            ref_intervals,ref_pitches,ref_velocities = evaluation.get_intervals_and_pitches_and_velocities(tempMid1)
            est_intervals,est_pitches,est_velocities = evaluation.get_intervals_and_pitches_and_velocities(asd)
            averageF1Score += mir_eval.transcription_velocity.precision_recall_f1_overlap(ref_intervals,ref_pitches,ref_velocities,est_intervals,est_pitches,est_velocities)[2]
            fileCounter += 1
    meanL2/= fileCounter
    averageF1Score /= fileCounter
    print("Average L2 score: " + str(meanL2))
    print("Average F1 Score: " + str(averageF1Score))
    print("Subset size: " + str(fileCounter))
    

if __name__ == '__main__':
    # genTestMidi()
    runTests()
    # runTestOnSubset()
    # directory = "/scratch2/lmd_tracks/lmd_tracks/d"
    # filename = 'd0ab3bfe7ac3936e1e184e11b516e5f3_5.mid'
    # f = os.path.join(directory, filename)
    # tempMid1 = pretty_midi.PrettyMIDI(f)
    # custom_plot_pianoroll(tempMid1)
    # thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    # asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    # custom_plot_pianoroll(asd)
    # est_intervals,est_pitches,est_velocities = evaluation.get_intervals_and_pitches_and_velocities(asd)