#!/usr/bin/env python3

import pypianoroll as pp
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import os
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False, help='Train instead of test')
args = parser.parse_args()

def get_data_files():
    data_files = []
    for dir_name, _, files in os.walk("/Users/dpagurek/Downloads/lpd_5/lpd_5_cleansed"):
        for file in files:
            if file.endswith(".npz"):
                data_files.append(f"{dir_name}/{file}")

    random.shuffle(data_files)

    return data_files

class BatchGenerator:
    def __init__(self, data_files, num_steps, batch_size, test_percent=0.2, test=False):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.test = test

        self.train_set = data_files[:int((1-test_percent)*len(data_files))]
        self.test_set = data_files[int((1-test_percent)*len(data_files)):]

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, 128))
        y = np.zeros((self.batch_size, self.num_steps, 128))
        generation_set = self.test_set if self.test else self.train_set
        while True:
            for data_file in generation_set:
                data = pp.load(data_file)
                raw_track = data.tracks[0].pianoroll
                step = 5
                track = np.array([ [ np.sign(np.sum(raw_track[i:i+step, j])) for j in range(128) ] for i in range(0, raw_track.shape[0]-1, step) ])
                # print(track)

                # 10 batches per sample track
                # for _ in range(10):
                if track.shape[0] < num_steps + 1:
                    continue

                for i in range(self.batch_size):
                    start_idx = np.random.randint(0, track.shape[0] - num_steps - 1)
                    x[i, :, :] = track[start_idx:start_idx+num_steps, :]
                    y[i, :, :] = track[start_idx+1:start_idx+1+num_steps, :]

                yield x, y

data_files = get_data_files()

num_steps = 200
batch_size = 20
num_epochs = 50
train_data_generator = BatchGenerator(data_files, num_steps, batch_size)
valid_data_generator = BatchGenerator(data_files, num_steps, batch_size, test=True)

hidden_size = 200
use_dropout = True
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(num_steps, 128)))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(128)))
model.add(Activation("softmax"))

optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

print(model.summary())

if args.train:
    checkpointer = ModelCheckpoint(filepath='./models/model-{epoch:02d}.hdf5', verbose=1)
    model.fit_generator(train_data_generator.generate(), 500, num_epochs,
            validation_data=valid_data_generator.generate(),
            validation_steps=100, callbacks=[checkpointer])
else:
    model = load_model("./models/model-01.hdf5")
    x, y = next(valid_data_generator.generate())
    prediction = model.predict([x[0]], batch_size=1)
    print(y[0])
    print(prediction)

    plt.imshow(prediction, cmap='hot', interpolation='nearest')
    plt.show()
