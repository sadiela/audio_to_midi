import mir_eval
import mido
import pretty_midi
import os


def createMidiToText(prettyMidi,tempo = 500000, filename = 'test.txt'):
    #The default tempo of midi files is 500000

    # testMid = mido.MidiFile(midiPath, clip=True)
    # for i in range(0,len(testMid.tracks[1])):
    #     print(testMid.tracks[1][i])

    #Create an array of note in hz, note on, note time
    listOfNoteTuples = []
    # time= 0
    # for msg in testMid.tracks[1]:
    #     if msg.type == "note_on":
    #         time+=msg.time
    #         if msg.velocity > 0:
    #             listOfNoteTuples.append((round(mir_eval.util.midi_to_hz(msg.note),2), "Down", mido.tick2second(time, testMid.ticks_per_beat,tempo)*100))
    #         else:
    #             listOfNoteTuples.append((round(mir_eval.util.midi_to_hz(msg.note),2), "Up", mido.tick2second(time, testMid.ticks_per_beat,tempo)*100))
    # print(listOfNoteTuples)
    for instrument in prettyMidi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                listOfNoteTuples.append((round(mir_eval.util.midi_to_hz(note.pitch),2), "Down", note.start*100))
                listOfNoteTuples.append((round(mir_eval.util.midi_to_hz(note.pitch),2), "Up", note.end*100))

    print(listOfNoteTuples)
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

def testmir_eval():
    ref_time, ref_freqs = mir_eval.io.load_ragged_time_series(
    'test.txt')
    est_time, est_freqs = mir_eval.io.load_ragged_time_series(
    'test.txt')
    results = mir_eval.multipitch.evaluate(ref_time,ref_freqs, est_time,est_freqs)
    print(results)

# createMidiToText(os.path.dirname(os.path.abspath(__file__)) + "/monophonic.mid", filename= "monophonic.txt")
# tempMid = pretty_midi.PrettyMIDI(os.path.dirname(os.path.abspath(__file__)) + "/monophonic.mid")
# createMidiToText(tempMid, filename= "monophonic.txt")

if __name__ == '__main__':
    testmir_eval()