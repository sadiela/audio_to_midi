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
            rounded_offset = round(note_offset,2)
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
            vocabEvent = event_dictionary[event]
            if "NOTE_ON:" in str(vocabEvent):
                currentNotes.append((event, currentTime))
            elif "NOTE_OFF:" in str(vocabEvent):
                for note_event in range(len(currentNotes)):
                    if "NOTE_ON:" in event_dictionary[currentNotes[note_event][0]]:
                        note = pretty_midi.Note(velocity=currentVelocity, pitch = currentNotes[note_event][0]-3, start = currentNotes[note_event][1], end = currentTime)
                        piano.notes.append(note)
                        currentNotes.remove(currentNotes[note_event])
                        break
            elif event >= 131 and event <= 631:
                currentTime = SEG_LENGTH_SECS*i + event_dictionary[event]
            elif "VEL:" in str(vocabEvent):
                currentVelocity = event-760
        
    created_midi.instruments.append(piano)
    return created_midi
                

if __name__ == '__main__':
    directory = '/scratch2/lmd_tracks/lmd_tracks/d'
    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         # tempMid = pretty_midi.PrettyMIDI(os.path.dirname(os.path.abspath(__file__)) + "/evaluation_test_data/complex.mid")
    #         tempMid = pretty_midi.PrettyMIDI(f)
    #         thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid)
    #         generatedMidi = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    #         generatedMidi.write(os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'+filename)
    
    gendirectory = os.path.dirname(os.path.abspath(__file__))+'/generatedmidi/'
    totalF1Score = 0
    numberOfFiles = 0
    filename = "evaluation_test_data/complex.mid"
    # filename = "evaluation_test_data/d0a0bf78d1d5a60ab02d08ee53f218a8_1.mid"
    # filename2 = "d0a0bf78d1d5a60ab02d08ee53f218a8_1.mid"
    # for filename in os.listdir(directory):
        
    
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    # f2 = os.path.join(directory, filename2)
    tempMid1 = pretty_midi.PrettyMIDI(f)
    # tempMid2 = pretty_midi.PrettyMIDI(f2)
    
    thing = pretty_midi_to_seq_chunks_w_noteoff_and_velocity(tempMid1)
    for chunk in thing:
        for i in chunk:
            if i == 1:
                break
            if i >130 and i < 632:
                print("TS: " + str(i)) 
            print(event_dictionary[i])
    
    asd = seq_chunks_w_noteoff_and_velocity_to_pretty_midi(thing)
    # asd.write(gendirectory +"d0a0bf78d1d5a60ab02d08ee53f218a8_1.mid")
    asd.write(gendirectory +"complex.mid")
    est_time, est_freqs = evaluation.createMidiToArray(asd)
    ref_time, ref_freqs = evaluation.createMidiToArray(tempMid1)
    with open("test1.txt", "w") as output:
        output.write(str(est_freqs))
    with open("test2.txt", "w") as output:
        output.write(str(ref_freqs))
    print(evaluation.evaluate(est_time, est_freqs,ref_time, ref_freqs))
    # est_time, est_freqs = evaluation.createMidiToArray(asd)
    # ref_time, ref_freqs = evaluation.createMidiToArray(tempMid2)
    # print(evaluation.evaluate(est_time, est_freqs,ref_time, ref_freqs))
    
    # # checking if it is a file
    # if os.path.isfile(f) and os.path.isfile(f2):
    #     tempMid1 = pretty_midi.PrettyMIDI(f)
    #     tempMid2 = pretty_midi.PrettyMIDI(f2)
    #     est_time, est_freqs = evaluation.createMidiToArray(tempMid1)
    #     ref_time, ref_freqs = evaluation.createMidiToArray(tempMid2)
    #     totalF1Score += evaluation.evaluate(est_time, est_freqs,ref_time, ref_freqs)
    #     numberOfFiles += 1
    # print("F1Score: "+ str(totalF1Score/numberOfFiles))


