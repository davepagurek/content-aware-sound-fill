#!/usr/bin/env python3

import pypianoroll as pp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
import math
import random
import os
import midi
import subprocess

DIRS = []
RANGE = (1, 1)
for x in range(-RANGE[1], RANGE[1]+1):
    for y in range(-RANGE[0], RANGE[0]+1):
        if x != 0 and y != 0:
            DIRS.append((x, y))
TILE_SIZE = (4, 2)

def generate_midi(data, offset, filename):
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
        for note in range(data.shape[1]):
            last_note_on = (step > 0 and data[step-1, note] == 1)
            note_on = (step < data.shape[0] and data[step, note] == 1)

            if note_on and not last_note_on:
                tick = 0
                if last != step:
                    tick = (step-last)*eigth_note
                    last = step

                track.append(midi.NoteOnEvent(tick=tick, velocity=90, pitch=note+offset))

            if not note_on and last_note_on:
                tick = 0
                if last != step:
                    tick = (step-last)*eigth_note
                    last = step

                track.append(midi.NoteOffEvent(tick=tick, pitch=note+offset))

    track.append(midi.EndOfTrackEvent(tick=eigth_note))
    midi.write_midifile(f"{filename}.mid", pattern)
    subprocess.call(["fluidsynth", "-F", f"{filename}.wav", "../lstm/GeneralUser GS MuseScore v1.442.sf2", f"{filename}.mid"])
    # subprocess.call(["rm", f"{filename}.mid"])
    subprocess.call(["open", f"{filename}.wav"])

class TrainingData:
    def __init__(self, data):
        self.data = data

    def check(self, a, b, direction):
        return (a, b, direction) in self.data

class WFC:
    def __init__(self, img, ignore):
        img = img[:img.shape[0]-(img.shape[0] % TILE_SIZE[0])][:img.shape[1]-(img.shape[1] % TILE_SIZE[1])]
        self.remaining_set = set(ignore)
        self.filled = []
        self.result = np.copy(img)
        for x, y in ignore:
            for xoff in range(TILE_SIZE[0]):
                for yoff in range(TILE_SIZE[1]):
                    self.result[y+yoff][x+xoff] = 0.5

        ignore_set = set(ignore)

        print("Computing training relationships")
        # Construct set of possible pixel relationships
        data = set()
        for y in range(RANGE[0]*TILE_SIZE[0], img.shape[0] - 2*RANGE[0]*TILE_SIZE[0], TILE_SIZE[0]):
            for x in range(RANGE[1]*TILE_SIZE[1], img.shape[1] - 2*RANGE[1]*TILE_SIZE[1], TILE_SIZE[1]):
                for off_y, off_x in DIRS:
                    if (x, y) in ignore_set:
                        continue
                    if (x+off_x*TILE_SIZE[1], y+off_y*TILE_SIZE[0]) in ignore_set:
                        continue

                    # Get pixels at the current location and off to the side
                    a = self.make_tile(x, y, img)
                    b = self.make_tile(x+off_x*TILE_SIZE[1], y+off_y*TILE_SIZE[0], img)
                    # a = (img[y][x][0], img[y][x][1], img[y][x][2])
                    # b = (img[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE][0], img[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE][1], img[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE][2])

                    # Record the occurrence of this relationship
                    data.add((a, b, (off_x, off_y)))
                    data.add((b, a, (-off_x, -off_y)))

        self.training_data = TrainingData(data)

        # Construct distribution of pixel values
        print("Computing distributions")
        all_pixels = []
        for y in range(0, img.shape[0] - TILE_SIZE[0], TILE_SIZE[0]):
            for x in range(0, img.shape[1] - TILE_SIZE[1], TILE_SIZE[1]):
                if (x, y) in self.remaining_set:
                    continue
                all_pixels.append(self.make_tile(x, y, img))
        pixel_counts = Counter(all_pixels)
        print(pixel_counts)
        total = sum([ pixel_counts[p] for p in pixel_counts.elements() ])

        self.probabilities = dict( (p, pixel_counts[p] / total) for p in pixel_counts.elements() )

        print("Computing initial entropies")
        self.remaining_entropy = dict( (pixel, self.entropy(pixel)) for pixel in self.remaining_set )

        print("Completed initialization")

    def make_tile(self, x, y, img):
        tile = []
        for yoff in range(TILE_SIZE[0]):
            row = []
            for xoff in range(TILE_SIZE[1]):
                row.append(img[y+yoff][x+xoff])
            tile.append(tuple(row))
        
        return tuple(tile)

    def fill(self, preview=lambda _: print("Milestone")):
        print("Filling")
        while len(self.remaining_set) > 0:
            print(f"{len(self.remaining_set)} remaining")
            self.collapse_one()
            if len(self.remaining_set) % 100 == 0:
                preview(self.result)

        
        return self.result

    def collapse_one(self):
        pixel = min(self.remaining_entropy, key=self.remaining_entropy.get)

        self.remaining_set.remove(pixel)
        self.remaining_entropy.pop(pixel)
        self.filled.append(pixel)

        options = self.options(pixel)

        if len(options) > 0:
            distribution = [ self.probabilities[option] for option in options ]
            total = sum(distribution)
            distribution = np.multiply(distribution, 1/total)
            collapsed = options[np.random.choice(range(len(options)), p=distribution)]
            print(collapsed)
            print(f"Collapsed {pixel} to value {collapsed}")
            for xoff in range(TILE_SIZE[1]):
                for yoff in range(TILE_SIZE[0]):
                    self.result[pixel[1]+yoff][pixel[0]+xoff] = collapsed[yoff][xoff]

            self.update_entropies_around(pixel)
        else:
            # num_to_backtrack = random.randint(1, len(self.filled))
            distribution = [math.exp(-x/200) for x in range(1, len(self.filled)+1)]
            distribution = np.multiply(distribution, 1/sum(distribution))
            num_to_backtrack = np.random.choice(range(1, len(self.filled)+1), p=distribution)
            # print(self.filled)
            print(f"Ran out of options to collapse, reverting {num_to_backtrack} choices")

            reverted = [pixel]# + self.filled[len(self.filled)-num_to_backtrack:]
            # self.filled = self.filled[:len(self.filled)-num_to_backtrack]

            for _ in range(num_to_backtrack):
                reverted_pixel = self.filled.pop(random.randint(0, len(self.filled)-1))
                reverted.append(reverted_pixel)

            for reverted_pixel in reverted:
                x, y = reverted_pixel
                for xoff in range(TILE_SIZE[1]):
                    for yoff in range(TILE_SIZE[0]):
                        self.result[y+yoff][x+xoff] = 0
                self.remaining_set.add(reverted_pixel)
                self.remaining_entropy[reverted_pixel] = self.entropy(reverted_pixel)

            for reverted_pixel in reverted:
                self.update_entropies_around(reverted_pixel)

            # print(self.filled)
            # plt.imshow(self.result, cmap='hot', interpolation='nearest')
            # plt.show()

    def update_entropies_around(self, pixel):
        x, y = pixel
        for off_x, off_y in DIRS:
            if y + off_y * TILE_SIZE[0] < 0 or y + off_y * TILE_SIZE[0] >= self.result.shape[0]:
                continue
            if x + off_x * TILE_SIZE[1] < 0 or x + off_x * TILE_SIZE[1] >= self.result.shape[1]:
                continue
            neighbour = (x + off_x * TILE_SIZE[1], y + off_y * TILE_SIZE[0])
            if neighbour in self.remaining_entropy:
                self.remaining_entropy[neighbour] = self.entropy(neighbour)

    def entropy(self, pixel):
        options = self.options(pixel)

        if len(options) == 0:
            return 0

        total_weight = 0
        total_weighted_logs = 0
        for option in options:
            weight = self.probabilities[option]
            total_weight += weight
            total_weighted_logs += weight * math.log2(weight)

        return math.log2(total_weight) - (total_weighted_logs / total_weight)

    def options(self, pixel):
        o = [ value for value in self.probabilities.keys() if self.is_option(pixel, value) ]
        return o

    def is_option(self, pixel, value):
        # num_ok = 0
        x, y = pixel
        for off_x, off_y in DIRS:
            if y + off_y*TILE_SIZE[0] < 0 or y + off_y*TILE_SIZE[0] >= self.result.shape[0]:
                continue
            if x + off_x*TILE_SIZE[1] < 0 or x + off_x*TILE_SIZE[1] >= self.result.shape[1]:
                continue
            if (x + off_x*TILE_SIZE[1], y + off_y*TILE_SIZE[0]) in self.remaining_set:
                continue

            # pixel = self.result[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE]
            # neighbour = (pixel[0], pixel[1], pixel[2])
            neighbour = self.make_tile(x+off_x*TILE_SIZE[1], y+off_y*TILE_SIZE[0], self.result)

            if not self.training_data.check(value, neighbour, (off_x, off_y)):
                return False

            # num_ok += 1
        return True

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


data_files = get_data_files()
track = None
while True:
    track = np.transpose(piano_track(random.choice(data_files)))
    if len(track) != 0:
        break

min_y = min(y for y in range(track.shape[0]) if sum(track[y]) > 0)
max_y = max(y for y in range(track.shape[0]) if sum(track[y]) > 0)
track = track[min_y:max_y+1]

print(track.shape)

predict_size = 60

start_idx = random.randint(0, track.shape[1] - predict_size)

imgplot = plt.imshow(track[:, start_idx-100:start_idx+predict_size+20], cmap='hot', interpolation='nearest')
plt.show()

to_fill = []
for y in range(0, track.shape[0] - TILE_SIZE[0], TILE_SIZE[0]):
    for x in range(0, track.shape[1] - TILE_SIZE[1], TILE_SIZE[1]):
        if start_idx <= x and x < start_idx + predict_size:
            to_fill.append((x, y))

def preview(filled):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].imshow(track[:, start_idx-100:start_idx+predict_size+20], cmap='hot', interpolation='nearest')
    ax[1].imshow(filled[:, start_idx-100:start_idx+predict_size+20], cmap='hot', interpolation='nearest')
    plt.show()

filled = WFC(track, to_fill).fill(preview)

print("done")

generate_midi(np.transpose(filled[:, start_idx-100:start_idx+predict_size+20]), min_y, "output")

preview(filled)
