from functions import decodeMidi
import mido
import numpy as np
import os

files = os.listdir('train')
features = np.zeros(shape=(1,1000,88))
labels = np.zeros(shape=(1,1000,88))
for file in files:
    print(file)
    mid = mido.MidiFile("train/"+file)
    mid = decodeMidi(mid)
    mid = mid[0][0]
    featuresPart = []
    labelsPart = []
    for i in range(1001):
        if i < len(mid):
            onehot = [0] * 88
            onehot[mid[i][1]] = 1
            featuresPart.append(onehot)
            onehot = [0] * 88
            try:
                onehot[mid[i+1][1]] = 1
                labelsPart.append(onehot)
            except Exception:
                pass
        else:
            pad = [0] * 88
            featuresPart.append(pad)
            labelsPart.append(pad)
    featuresPart = np.array([featuresPart[:1000]])
    labelsPart = np.array([labelsPart[:1000]])
    features = np.concatenate((features,featuresPart))
    labels = np.concatenate((labels,labelsPart))
    print(featuresPart.shape)
    print(labelsPart.shape)

features = features[1:]
labels = labels[1:]
print(features.shape)
print(labels.shape)


from keras.models import Sequential
from keras.layers import GRU, TimeDistributed, RepeatVector, Flatten, Embedding, LSTM, Dropout, Dense, Activation, MaxPooling1D, LeakyReLU, SimpleRNN
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(SimpleRNN(input_dim  =  88, output_dim = 88, return_sequences = True))
model.add(TimeDistributed(Dense(output_dim = 88, activation  =  "softmax")))
model.compile(loss = "mse", optimizer = "rmsprop", metrics=['accuracy'])
model.fit(features, labels,
          epochs=1000,
          batch_size=256,
          callbacks=[ModelCheckpoint("Simple_RNN_3", monitor='val_acc',save_best_only=True)])
model.save("Simple_RNN_3.h5")
