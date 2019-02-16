#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
import math
from random import randint

UP = (0, -1)
LEFT = (-1, 0)
DOWN = (0, 1)
RIGHT = (1, 0)
# DIRS = [UP, DOWN, LEFT, RIGHT]
DIRS = []
RANGE = 4
for x in range(-RANGE, RANGE+1):
    for y in range(-RANGE, RANGE+1):
        if x != 0 and y != 0:
            DIRS.append((x, y))
TILE_SIZE = 2

class TrainingData:
    def __init__(self, data):
        self.data = data

    def check(self, a, b, direction):
        return (a, b, direction) in self.data

class WFC:
    def __init__(self, img, ignore):
        img = img[:img.shape[1]-(img.shape[1] % TILE_SIZE)][:img.shape[0]-(img.shape[0] % TILE_SIZE)]
        self.remaining_set = set(ignore)
        self.filled = []
        self.result = np.copy(img)
        for x, y in ignore:
            for xoff in range(TILE_SIZE):
                for yoff in range(TILE_SIZE):
                    for channel in range(3):
                        self.result[y+yoff][x+xoff][channel] = 0

        ignore_set = set(ignore)

        print("Computing training relationships")
        # Construct set of possible pixel relationships
        data = set()
        for y in range(RANGE*TILE_SIZE, img.shape[0] - RANGE*TILE_SIZE, TILE_SIZE):
            for x in range(RANGE*TILE_SIZE, img.shape[1] - RANGE*TILE_SIZE, TILE_SIZE):
                for off_x, off_y in DIRS:
                    if (x, y) in ignore_set:
                        continue
                    if (x+off_x*TILE_SIZE, y+off_y*TILE_SIZE) in ignore_set:
                        continue

                    # Get pixels at the current location and off to the side
                    a = self.make_tile(x, y, img)
                    b = self.make_tile(x+off_x*TILE_SIZE, y+off_y*TILE_SIZE, img)
                    # a = (img[y][x][0], img[y][x][1], img[y][x][2])
                    # b = (img[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE][0], img[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE][1], img[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE][2])

                    # Record the occurrence of this relationship
                    data.add((a, b, (off_x, off_y)))
                    data.add((b, a, (-off_x, -off_y)))

        self.training_data = TrainingData(data)

        # Construct distribution of pixel values
        print("Computing distributions")
        all_pixels = []
        for y in range(0, img.shape[0] - TILE_SIZE, TILE_SIZE):
            for x in range(0, img.shape[1] - TILE_SIZE, TILE_SIZE):
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
        for yoff in range(TILE_SIZE):
            row = []
            for xoff in range(TILE_SIZE):
                channels = img[y+yoff][x+xoff]
                row.append((channels[0], channels[1], channels[2]))
            tile.append(tuple(row))
        
        return tuple(tile)

    def fill(self):
        print("Filling")
        while len(self.remaining_set) > 0:
            print(f"{len(self.remaining_set)} remaining")
            self.collapse_one()
            if len(self.remaining_set) % 100 == 0:
                print(len(self.remaining_set))
                plt.imshow(self.result)
                plt.show()

        
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
            for xoff in range(TILE_SIZE):
                for yoff in range(TILE_SIZE):
                    for channel in range(3):
                        self.result[pixel[1]+yoff][pixel[0]+xoff][channel] = collapsed[yoff][xoff][channel]
                    self.result[pixel[1]+yoff][pixel[0]+xoff][3] = 1

            self.update_entropies_around(pixel)
        else:
            # num_to_backtrack = randint(1, len(self.filled))
            distribution = [math.exp(-x/5) for x in range(1, len(self.filled)+1)]
            distribution = np.multiply(distribution, 1/sum(distribution))
            num_to_backtrack = np.random.choice(range(1, len(self.filled)+1), p=distribution)
            print(self.filled)
            print(f"Ran out of options to collapse, reverting {num_to_backtrack} choices")

            reverted = [pixel]# + self.filled[len(self.filled)-num_to_backtrack:]
            # self.filled = self.filled[:len(self.filled)-num_to_backtrack]

            for _ in range(num_to_backtrack):
                reverted_pixel = self.filled.pop(randint(0, len(self.filled)-1))
                reverted.append(reverted_pixel)

            for reverted_pixel in reverted:
                x, y = reverted_pixel
                for xoff in range(TILE_SIZE):
                    for yoff in range(TILE_SIZE):
                        for channel in range(3):
                            self.result[y+yoff][x+xoff][channel] = 0
                self.remaining_set.add(reverted_pixel)
                self.remaining_entropy[reverted_pixel] = self.entropy(reverted_pixel)

            for reverted_pixel in reverted:
                self.update_entropies_around(reverted_pixel)

            print(self.filled)
            plt.imshow(self.result)
            plt.show()

    def update_entropies_around(self, pixel):
        x, y = pixel
        for off_x, off_y in DIRS:
            if y + off_y * TILE_SIZE < 0 or y + off_y * TILE_SIZE >= self.result.shape[0]:
                continue
            if x + off_x * TILE_SIZE < 0 or x + off_x * TILE_SIZE >= self.result.shape[1]:
                continue
            neighbour = (x + off_x * TILE_SIZE, y + off_y * TILE_SIZE)
            if neighbour in self.remaining_entropy:
                self.remaining_entropy[neighbour] = self.entropy(neighbour)

    def entropy(self, pixel):
        options = self.options(pixel)

        if len(options) == 0:
            return math.inf

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
            if y + off_y*TILE_SIZE < 0 or y + off_y*TILE_SIZE >= self.result.shape[0]:
                continue
            if x + off_x*TILE_SIZE < 0 or x + off_x*TILE_SIZE >= self.result.shape[1]:
                continue
            if (x + off_x*TILE_SIZE, y + off_y*TILE_SIZE) in self.remaining_set:
                continue

            # pixel = self.result[y+off_y*TILE_SIZE][x+off_x*TILE_SIZE]
            # neighbour = (pixel[0], pixel[1], pixel[2])
            neighbour = self.make_tile(x+off_x*TILE_SIZE, y+off_y*TILE_SIZE, self.result)

            if not self.training_data.check(value, neighbour, (off_x, off_y)):
                return False

            # num_ok += 1
        return True


img = mpimg.imread("examples/lines.png")
img = np.multiply(np.round(np.multiply(img, 1)), 1/1)
imgplot = plt.imshow(img)
plt.show()

to_fill = []
for y in range(0, img.shape[0] - TILE_SIZE, TILE_SIZE):
    for x in range(0, img.shape[1] - TILE_SIZE, TILE_SIZE):
        if 255 <= x and x < 283:
            to_fill.append((x, y))
filled = WFC(img, to_fill).fill()

print("done")
filledplot = plt.imshow(filled)

plt.show()
