import pickle
from functions import predictionsToNotes, notesToMidi
import mido

pred = pickle.load(open('moonlightPrediction.pickle','rb'))
pred = predictionsToNotes(pred)
print(pred)
file = notesToMidi(pred,time=60)
print(file)


# decodedSong = encodeMidi(newSong)

# data = decodedSong
file.ticks_per_beat = 120
port = mido.open_output()

# for i, track in enumerate(data.tracks):
#     print('Track {}: {}'.format(i, track.name))
#     for msg in track:
#         print(msg)

for msg in file.play():
    port.send(msg)

file.save("monnlight_sonata_bach.mid")