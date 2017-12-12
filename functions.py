import numpy as np
import mido

def decodeMidi(midifile,num_layers=2):
    song = []
    for i, track in enumerate(midifile.tracks):
        song.append([])
        for msg in track:
            message = str(msg).split()
            if '<meta' not in message and 'control_change' not in message and 'program_change' not in message:
                channel = int(str(msg).split()[1].split("=")[-1])
                note = int(str(msg).split()[2].split("=")[-1])
                velocity = int(str(msg).split()[3].split("=")[-1])
                time = int(str(msg).split()[4].split('=')[-1])
                song[i].append([channel, note, velocity, time])

    song = [x for x in song if x]
    song = [song[:num_layers]]
    return np.array(song)

def encodeMidi(song):
    file = mido.MidiFile()
    for i in range(len(song[0])):
        track = mido.MidiTrack()
        track.append(mido.Message('control_change', channel=0, control=0, value=80, time=0))
        track.append(mido.Message('control_change', channel=0, control=32, value=0, time=0))
        track.append(mido.Message('program_change', channel=0, program=50, time=0))
        for j in range(len(song[0][i])):
            note = mido.Message('note_on',channel=int(song[0][i][j][0]), note=int(song[0][i][j][1]), velocity=int(song[0][i][j][2]), time=int(song[0][i][j][3]))
            track.append(note)
        file.tracks.append(track)
    return file


def midiToNote(midifile,num_layers=2):
    song = []
    for i, track in enumerate(midifile.tracks):
        song.append([])
        for msg in track:
            message = str(msg).split()
            if '<meta' not in message and 'control_change' not in message and 'program_change' not in message:
                note = int(str(msg).split()[2].split("=")[-1])
                song[i].append([note])

    song = [x for x in song if x]
    song = [song[:num_layers]]
    return np.array(song)

def predictionsToNotes(preds):
    song = preds[0][0]
    song = song.tolist()
    _ = 0
    noteIndex = []
    for i in song:
        best = max(song[_])
        key = song[_].index(best)
        noteIndex.append(key)
        # print(best,key)
        _ += 1
    return (noteIndex)

def notesToMidi(notes, velocity = 95, time = 116):
    file = mido.MidiFile()
    track = mido.MidiTrack()
    file.tracks.append(track)
    track.append(mido.Message('program_change', program=12, time=time))
    for i in range(len(notes)):
        track.append(mido.Message('note_on', note=notes[i], velocity=velocity, time=time))
    return(file)

def timeAndNotesToMidi(notes,timeList,velocity = 95):
    file = mido.MidiFile()
    track = mido.MidiTrack()
    file.tracks.append(track)
    track.append(mido.Message('program_change', program=12, time=116))
    for i in range(len(timeList)):
        track.append(mido.Message('note_on', note=notes[i], velocity=velocity, time=timeList[i]))
        print(notes[i],timeList[i])
    return (file)

def midiToTime(midifile, num_layers=2):
    song = []
    for i, track in enumerate(midifile.tracks):
        song.append([])
        for msg in track:
            message = str(msg).split()
            if '<meta' not in message and 'control_change' not in message and 'program_change' not in message:
                time = int(str(msg).split()[4].split("=")[-1])
                song[i].append([time])

    song = [x for x in song if x]
    song = [song[:num_layers]]
    return song

# Jesu = mido.MidiFile("D:/Downloads/JESU1.mid")
# BSG = mido.MidiFile("D:/Downloads/BSG.mid")
# #
# jesuDec = decodeMidi(Jesu,num_layers=2)
# bsgDec = decodeMidi(BSG,num_layers=2)
# print(jesuDec.shape)
# print(bsgDec.shape)
# #
# # features = np.concatenate((jesuDec,bsgDec))
# # print(features.shape)
# jesuNotes = midiToNote(Jesu,num_layers=1)
# bsgNotes = midiToNote(BSG,num_layers=1)
# print(jesuNotes.shape)
# print(bsgNotes.shape)