#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import Counter
import math

UP = (0, 1)
LEFT = (-1, 0)
DOWN = (0, -1)
RIGHT = (1, 0)
DIRS = [UP, DOWN, LEFT, RIGHT]

class TrainingData:
    def __init__(self, data):
        self.data = data

    def check(self, a, b, direction):
        return (a, b, direction) in self.data

class WFC:
    def __init__(self, img, ignore):
        self.remaining_set = set(ignore)
        self.filled = []
        self.result = np.copy(img)

        ignore_set = set(ignore)

        print("Computing training relationships")
        # Construct set of possible pixel relationships
        data = set()
        for y in range(1, img.shape[0] - 1):
            for x in range(1, img.shape[1] - 1):
                for off_x, off_y in DIRS:
                    if (x, y) in ignore_set:
                        continue
                    if (x+off_x, y+off_y) in ignore_set:
                        continue

                    # Get pixels at the current location and off to the side
                    a = (img[y][x][0], img[y][x][1], img[y][x][2])
                    b = (img[y+off_y][x+off_x][0], img[y+off_y][x+off_x][1], img[y+off_y][x+off_x][2])

                    # Record the occurrence of this relationship
                    data.add((a, b, (off_x, off_y)))
                    data.add((b, a, (-off_x, -off_y)))

        self.training_data = TrainingData(data)

        # Construct distribution of pixel values
        print("Computing distributions")
        all_pixels = []
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if (x, y) in self.remaining_set:
                    continue
                all_pixels.append((img[y][x][0], img[y][x][1], img[y][x][2]))
        pixel_counts = Counter(all_pixels)
        total = sum([ pixel_counts[p] for p in pixel_counts.elements() ])

        self.probabilities = dict( (p, pixel_counts[p] / total) for p in pixel_counts.elements() )

        print("Computing initial entropies")
        self.remaining_entropy = dict( (pixel, self.entropy(pixel)) for pixel in self.remaining_set )

        print("Completed initialization")

    def fill(self):
        print("Filling")
        while len(self.remaining_set) > 0:
            print(f"{len(self.remaining_set)} remaining")
            self.collapse_one()
            if len(self.remaining_set) % 20 == 0:
                print(len(self.remaining_set))
                plt.imshow(self.result)
                plt.show()

        
        return self.result

    def collapse_one(self):
        pixel = min(remaining_entropy, key=remaining_entropy.get)

        self.remaining_set.remove(pixel)
        self.remaining_entropy.remove(pixel)
        self.filled.push(pixel)

        options = self.options(pixel)

        if len(options) > 0:
            distribution = [ self.probabilities[option] for option in options ]
            collapsed = np.random.choice(options, p=distribution)
            print(f"Collapsed {pixel} to value {collapsed}")
            for channel in range(3):
                result[pixel[1]][pixel[0]][channel] = collapsed[channel]
            result[pixel[1]][pixel[0]][3] = 1

            self.update_entropies_around(pixel)
        else:
            # TODO try backtracking
            raise ValueError("Ran out of options to collapse!")

    def update_entropies_around(self, pixel):
        x, y = pixel
        for off_x, off_y in DIRS:
            if y + off_y < 0 or y + off_y >= self.result.shape[0]:
                continue
            if x + off_x < 0 or x + off_x >= self.result.shape[1]:
                continue

            neighbour = (x + off_x, y + off_y)
            self.remaining_entropy[neighbour] = self.entropy(neighbour)

    def entropy(self, pixel):
        options = self.options(pixel)

        total_weight = 0
        total_weighted_logs = 0
        for option in options:
            weight = self.probabilities[option]
            total_weight += weight
            total_weighted_logs += weight * math.log2(weight)

        return math.log2(total_weight) - (total_weighted_logs / total_weight)

    def options(self, pixel):
        print(f"Filtering {len(self.probabilities)} options for pixel {pixel}")
        return [ value for value in self.probabilities.keys() if self.is_option(pixel, value) ]

    def is_option(self, pixel, value):
        for off_x, off_y in DIRS:
            if y + off_y < 0 or y + off_y >= self.result.shape[0]:
                continue
            if x + off_x < 0 or x + off_x >= self.result.shape[1]:
                continue
            if (x + off_x, y + off_y) in self.remaining_set:
                continue

            pixel = self.result[y+off_y][x+off_x]
            neighbour = (pixel[0], pixel[1], pixel[2])

            if not self.training_data.check(value, neighbour, (off_x, off_y)):
                return False

        return True


img = mpimg.imread("examples/landscape.png")
imgplot = plt.imshow(img)

to_fill = []
for x in range(255, 283):
    for y in range(101, 154):
        to_fill.append((x, y))
filled = WFC(img, to_fill).fill()
filledplot = plt.imshow(filled)

plt.show()
