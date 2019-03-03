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
import midi
import subprocess
import itertools

p = midi.Pattern(resolution=96/2)

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=False, help='Train instead of test')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=False, help='Evaluate instead of test')
parser.add_argument('--evaluate_many', dest='evaluate_many', action='store_true', default=False, help='Evaluate across multiple songs')
args = parser.parse_args()

def generate_midi(data, filename):
    eigth_note = 96//2

    instrument = 1
    # 1 - grand piano
    # 5 - electric piano
    # 41 - violin
    # 27 - electric guitar

    pattern = midi.Pattern(resolution=eigth_note)
    track = midi.Track()
    track.append(midi.ProgramChangeEvent(tick=0, data=[instrument]))
    pattern.append(track)

    set_tempo_event = midi.SetTempoEvent(tick=0)
    set_tempo_event.bpm = 200
    track.append(set_tempo_event)

    last = 0
    for step in range(data.shape[0] + 1):
        for note in range(128):
            last_note_on = (step > 0 and data[step-1, note] == 1)
            note_on = (step < data.shape[0] and data[step, note] == 1)

            if note_on and not last_note_on:
                tick = 0
                if last != step:
                    tick = (step-last)*eigth_note
                    last = step

                track.append(midi.NoteOnEvent(tick=tick, velocity=90, pitch=note))

            if not note_on and last_note_on:
                tick = 0
                if last != step:
                    tick = (step-last)*eigth_note
                    last = step

                track.append(midi.NoteOffEvent(tick=tick, pitch=note))

    track.append(midi.EndOfTrackEvent(tick=eigth_note))
    midi.write_midifile(f"{filename}.mid", pattern)
    subprocess.call(["fluidsynth", "-F", f"{filename}.wav", "GeneralUser GS MuseScore v1.442.sf2", f"{filename}.mid"])
    # subprocess.call(["rm", f"{filename}.mid"])
    subprocess.call(["open", f"{filename}.wav"])

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
    step = 4
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
# model.add(Activation("softmax"))

optimizer = Adam()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print(model.summary())

def probability(prediction):
    bias = -0.065
    scale = 1.3
    return max(0, min(1, prediction * scale + bias))

def make_sl(likelihoods, cost):
    # print(np.prod(np.multiply(likelihoods, math.pow(1e9, 1/len(likelihoods)))) / math.exp(cost))
    return np.prod(np.multiply(likelihoods, math.pow(1e20, 1/len(likelihoods)))) / math.exp(cost)

def distances(data):
    d = []
    for step in range(data.shape[0]):
        sounded = [ note for note in range(128) if data[step, note] == 1 ]
        for a, b in itertools.combinations(sounded, 2):
            d.append(abs(a-b) % 12)
    return d

def predict(model, track, samples=20, predict_size=60, suffix_size=20, nonempty=False):
    start_idx = None
    while start_idx is None or (nonempty and sum(distances(track[start_idx+num_steps:start_idx+num_steps+predict_size])) == 0):
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
            data[0, i+num_steps, note] = 1 if random.uniform(0, 1) < probability(prediction[note]) else 0

        likelihoods.append(np.prod([ probability(prediction[note]) if data[0, i+num_steps, note] == 1 else 1 - probability(prediction[note]) for note in range(128) ]))

    verification = model.predict(data[0:1, -num_steps:, :], batch_size=1)
    cost = 0
    for i in range(suffix_size - 1):
        for j in range(128):
            cost += (verification[0, -i, j] - data[0, -i-1, j])**2
    sl = make_sl(likelihoods, cost)

    for _ in range(samples):
        distribution = [math.exp(-x/5) for x in range(1, 1+predict_size)]
        distribution = np.multiply(distribution, 1/sum(distribution))
        backtrack_length = np.random.choice(range(1, 1+predict_size), p=distribution)

        candidate_likelihoods = likelihoods[:-backtrack_length]
        candidate_data = np.copy(data)

        for i in range(predict_size-backtrack_length, predict_size):
            output = model.predict(data[0:1, i:i+num_steps, :], batch_size=1)
            prediction = output[0,-1,:]

            for note in range(128):
                candidate_data[0, i+num_steps, note] = 1 if random.uniform(0, 1) < probability(prediction[note]) else 0

            candidate_likelihoods.append(np.prod([ probability(prediction[note]) if candidate_data[0, i+num_steps, note] == 1 else 1 - probability(prediction[note]) for note in range(128) ]))

        verification = model.predict(candidate_data[0:1, -num_steps:, :], batch_size=1)
        candidate_cost = 0
        for i in range(suffix_size - 1):
            for j in range(128):
                candidate_cost += (verification[0, -i, j] - candidate_data[0, -i-1, j])**2

        candidate_sl = make_sl(candidate_likelihoods, candidate_cost)

        print(f"{sl} vs {candidate_sl}")
        if candidate_sl > sl or random.uniform(0, 1) < candidate_sl / sl:
            print(f"Updating to: {candidate_cost}")
            cost = candidate_cost
            data = candidate_data
            likelihoods = candidate_likelihoods
            sl = candidate_sl

    print(cost)

    filled = (num_steps, num_steps + predict_size)
    return data[0], track[start_idx:start_idx+num_steps+predict_size+suffix_size], filled

if args.train:
    checkpointer = ModelCheckpoint(filepath='./models/model-{epoch:02d}.hdf5', verbose=1)
    model.fit_generator(train_data_generator.generate(), 500, num_epochs,
            validation_data=valid_data_generator.generate(),
            validation_steps=100, callbacks=[checkpointer])
elif args.evaluate:
    model = load_model("./models/model-09.hdf5")
    track = None

    predict_size = 60
    suffix_size = 20

    while track is None or track.shape[0] <= predict_size + num_steps + suffix_size:
        track = piano_track(random.choice(data_files))

    real_distances = []
    filled_distances = []
    song_distances = []
    for _ in range(3):
        data, truth, (start, end) = predict(model, track, predict_size=predict_size, suffix_size=suffix_size)
        real_distances.extend(distances(truth[start:end,:]))
        filled_distances.extend(distances(data[start:end, :]))
        song_distances.extend(distances(truth[:start, :]))
        song_distances.extend(distances(truth[end:, :]))

    print(f"Real: {sum(real_distances)}")
    print(f"Filled: {sum(filled_distances)}")
    bins = list(range(13))
    ticks = [ x+0.5 for x in bins ]
    intervals = ["P1", "m2", "M2", "m3", "M3", "P4", "d5", "P5", "m6", "M6", "m7", "M7"]
    plt.hist([real_distances, filled_distances, song_distances], bins, density=True, label=["Ground truth", "Filled", "Rest of song"])
    plt.xlim(0, 12)
    plt.xticks(ticks, intervals)
    plt.title("Interval Distribution")
    plt.xlabel("Interval")
    plt.ylabel("Percent of intervals")
    plt.legend(loc='upper right')
    plt.show()

elif args.evaluate_many:
    model = load_model("./models/model-09.hdf5")

    truth_errors = []
    filled_errors = []

    while len(truth_errors) < 15:
        track = None

        predict_size = 60
        suffix_size = 20

        while track is None or track.shape[0] <= predict_size + num_steps + suffix_size:
            track = piano_track(random.choice(data_files))

        real_distances = []
        filled_distances = []
        song_distances = []

        data, truth, (start, end) = predict(model, track, predict_size=predict_size, suffix_size=suffix_size, nonempty=True)

        real_distances.extend(distances(truth[start:end,:]))
        filled_distances.extend(distances(data[start:end, :]))
        song_distances.extend(distances(truth[:start, :]))
        song_distances.extend(distances(truth[end:, :]))

        bins = list(range(13))
        real_hist, _ = np.histogram(real_distances, bins=bins, density=True)
        filled_hist, _ = np.histogram(filled_distances, bins=bins, density=True)
        song_hist, _ = np.histogram(song_distances, bins=bins, density=True)

        truth_error = sum([ (a-b)**2 for a, b in zip(real_hist, song_hist) ])
        filled_error = sum([ (a-b)**2 for a, b in zip(filled_hist, song_hist) ])
        if (not math.isnan(truth_error)) and (not math.isnan(filled_error)):
            truth_errors.append(truth_error)
            filled_errors.append(filled_error)

    print("Making boxplot")
    print(truth_errors)
    print(filled_errors)
    plt.boxplot([ truth_errors, filled_errors ], labels=["Truth to song", "Filled to song"])
    plt.show()

else:
    model = load_model("./models/model-09.hdf5")
    track = None

    predict_size = 60
    suffix_size = 20

    while track is None or track.shape[0] <= predict_size + num_steps + suffix_size:
        track = piano_track(random.choice(data_files))

    for _ in range(4):
        data, truth, _ = predict(model, track, predict_size=predict_size, suffix_size=suffix_size)

        # print(y[0])
        # print(prediction[0])
        # print(sum(prediction[0][-1]))

        generate_midi(data, "output")

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        ax[0].imshow(np.transpose(truth), cmap='hot', interpolation='nearest')
        ax[1].imshow(np.transpose(data), cmap='hot', interpolation='nearest')
        plt.show()
