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
import math
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

def piano_track(data_file):
    data = pp.load(data_file)
    index = [track.name for track in data.tracks].index("Piano")
    raw_track = data.tracks[index].pianoroll
    step = 5
    track = np.array([ [ np.sign(np.sum(raw_track[i:i+step, j])) for j in range(128) ] for i in range(0, raw_track.shape[0]-1, step) ])

    return track

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
                track = piano_track(data_file)
                # print(track)

                # 10 batches per sample track
                for _ in range(10):
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
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print(model.summary())

if args.train:
    checkpointer = ModelCheckpoint(filepath='./models/model-{epoch:02d}.hdf5', verbose=1)
    model.fit_generator(train_data_generator.generate(), 500, num_epochs,
            validation_data=valid_data_generator.generate(),
            validation_steps=100, callbacks=[checkpointer])
else:
    model = load_model("./models/model-07.hdf5")
    track = None

    predict_size = 60
    suffix_size = 20

    bias = 0
    scale = 2

    while track is None or track.shape[0] <= predict_size + num_steps + suffix_size:
        track = piano_track(random.choice(data_files))


    for _ in range(4):
        start_idx = random.randint(0, track.shape[0] - predict_size - num_steps - suffix_size)

        data = np.zeros((1, num_steps + predict_size + suffix_size, 128))
        # add prefix
        data[0:1, 0:num_steps, :] = track[start_idx:start_idx+num_steps, :]
        # add suffix
        data[0:1, num_steps+predict_size:, :] = track[start_idx+num_steps+predict_size:start_idx+num_steps+predict_size+suffix_size, :]
        likelihoods = []

        for i in range(predict_size):
            output = model.predict(data[0:1, i:i+num_steps, :], batch_size=1)
            prediction = output[0,-1,:]

            for note in range(128):
                probability = min(1, prediction[note] * scale + bias)
                data[0, i+num_steps, note] = 1 if random.uniform(0, 1) < probability else 0

            likelihoods.append(np.prod([ probability if data[0, i+num_steps, note] == 1 else 1 - probability for note in range(128) ]))

        verification = model.predict(data[0:1, -num_steps:, :], batch_size=1)
        cost = 0
        for i in range(suffix_size - 1):
            for j in range(128):
                cost += (verification[0, -i, j] - data[0, -i-1, j])**2
        sl = np.prod(np.multiply(likelihoods, 1e7)) / math.exp(cost)

        for _ in range(20):
            distribution = [math.exp(-x/5) for x in range(1, 1+predict_size)]
            distribution = np.multiply(distribution, 1/sum(distribution))
            backtrack_length = np.random.choice(range(1, 1+predict_size), p=distribution)

            candidate_likelihoods = likelihoods[:-backtrack_length]
            candidate_data = np.copy(data)

            for i in range(predict_size-backtrack_length, predict_size):
                output = model.predict(data[0:1, i:i+num_steps, :], batch_size=1)
                prediction = output[0,-1,:]

                for note in range(128):
                    probability = min(1, prediction[note] * scale + bias)
                    candidate_data[0, i+num_steps, note] = 1 if random.uniform(0, 1) < probability else 0

                candidate_likelihoods.append(np.prod([ probability if candidate_data[0, i+num_steps, note] == 1 else 1 - probability for note in range(128) ]))

            verification = model.predict(candidate_data[0:1, -num_steps:, :], batch_size=1)
            candidate_cost = 0
            for i in range(suffix_size - 1):
                for j in range(128):
                    candidate_cost += (verification[0, -i, j] - candidate_data[0, -i-1, j])**2

            candidate_sl = np.prod(np.multiply(candidate_likelihoods, 1e7)) / math.exp(candidate_cost)

            print(f"{sl:.8f} vs {candidate_sl:.8f}")
            if candidate_sl > sl or random.uniform(0, 1) < candidate_sl / sl:
                print(f"Updating to: {candidate_cost}")
                cost = candidate_cost
                data = candidate_data
                likelihoods = candidate_likelihoods
                sl = candidate_sl

        print(cost)




        # print(y[0])
        # print(prediction[0])
        # print(sum(prediction[0][-1]))

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].imshow(np.transpose(track[start_idx:start_idx+num_steps+predict_size+suffix_size]), cmap='hot', interpolation='nearest')
        ax[1].imshow(np.transpose(data[0]), cmap='hot', interpolation='nearest')
        plt.show()
