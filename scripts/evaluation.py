import mir_eval
import mido
import pretty_midi
import os
import numpy as np


def createMidiToText(prettyMidi, filename = 'test.txt'):

    listOfNoteTuples = []

    for instrument in prettyMidi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                listOfNoteTuples.append((round(mir_eval.util.midi_to_hz(note.pitch),2), "Down", note.start*100))
                listOfNoteTuples.append((round(mir_eval.util.midi_to_hz(note.pitch),2), "Up", note.end*100))

    # print(listOfNoteTuples)
    # print(testMid.length)
    currentNotes = []
    
    f = open(os.path.dirname(os.path.abspath(__file__))+"/"+filename, "w")
    for timeStep in range(0,int(round(prettyMidi.get_end_time(),2)*100)):
        f.write(str(timeStep/100.0))
        for note in listOfNoteTuples:
            if note[1] == 'Down' and int(note[2]) == timeStep and note[0] not in currentNotes:
                currentNotes.append(note[0])
            if note[1] == 'Up' and int(note[2]) == timeStep and note[0] in currentNotes:
                currentNotes.remove(note[0])
        for note in currentNotes:
            f.write(" " + str(note))
        f.write("\n")
    f.close()


def createMidiToArray(prettyMidi):

    currentNotes = []

    est_time = np.arange(0,int(round(prettyMidi.get_end_time(),2)), 0.01)
    est_freqs = []
    piano_roll = prettyMidi.get_piano_roll(100,None,None).transpose()

    for timeStep in range(0,len(est_time)):
        for freq in range(len(piano_roll[timeStep])):
            if not piano_roll[timeStep][freq] == 0:
                currentNotes.append(round(mir_eval.util.midi_to_hz(freq),2))
        est_freqs.append(np.array(currentNotes))
        currentNotes = []
    return (est_time,est_freqs)
        
    

def evaluateFromFile():
    ref_time, ref_freqs = mir_eval.io.load_ragged_time_series(
    os.path.dirname(os.path.abspath(__file__)) + '/evaluation_test_data/test.txt')
    est_time, est_freqs = mir_eval.io.load_ragged_time_series(
    os.path.dirname(os.path.abspath(__file__)) + '/evaluation_test_data/test.txt')
    results = mir_eval.multipitch.evaluate(ref_time,ref_freqs, est_time,est_freqs)

    # print f1 scores
    print(type(ref_freqs))
    print(getF1Score(results['Precision'], results['Recall']))

def evaluateRefFile(est_time,est_freqs):
    ref_time, ref_freqs = mir_eval.io.load_ragged_time_series(
    os.path.dirname(os.path.abspath(__file__)) + '/evaluation_test_data/monophonic.txt')
    results = mir_eval.multipitch.evaluate(ref_time,ref_freqs, est_time,est_freqs)

    # print f1 scores
    return getF1Score(results['Precision'], results['Recall'])

def evaluate(est_time,est_freqs, ref_time, ref_freqs):
    results = mir_eval.multipitch.evaluate(ref_time,ref_freqs, est_time,est_freqs)
    return getF1Score(results['Precision'], results['Recall'])
    
def getF1Score(precision, recall):
    return 2 * (precision*recall)/(precision+recall)




# createMidiToText(os.path.dirname(os.path.abspath(__file__)) + "/monophonic.mid", filename= "monophonic.txt")


if __name__ == '__main__':
    directory = '/scratch2/lmd_tracks/lmd_tracks/d/d0a3ad8cb51e2bf621b9fc3ff0ea155e_8.mid'
    tempMid = pretty_midi.PrettyMIDI(os.path.dirname(os.path.abspath(__file__)) + "/evaluation_test_data/monophonic.mid")
    tempMid = pretty_midi.PrettyMIDI(directory)
    est_time, est_freq = createMidiToArray(tempMid)
    # print(est_time)
    # print(est_freq)
    print(evaluate(est_time, est_freq,est_time, est_freq))