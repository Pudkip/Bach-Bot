from keras.models import load_model
from functions import decodeMidi
import mido
import numpy as np
import pickle

song = mido.MidiFile("moonlightSonata.mid")
song = decodeMidi(song)
song = song[0][0]
print(song)
test = []

for i in range(len(song)):
    onehot = [0] * 88
    onehot[song[i][1]] = 1
    test.append(onehot)
    onehot = [0] * 88
    # try:
    #     onehot[Jesu[i+1][1]] = 1
    #     labels.append(onehot)
    # except Exception:
    #     pass
test = np.array([test[:1000]])
print(test.shape)

model = load_model('Simple_RNN_3.h5')
prediction = [model.predict(test)]

savePredictions = open("moonlightPrediction.pickle","wb")
pickle.dump(prediction, savePredictions)
savePredictions.close()