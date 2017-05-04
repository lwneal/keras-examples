import time
import os
import random
from imutil import show, decode_jpg

from keras import models, layers, applications
import numpy as np

# Set this to a smaller number to run faster
DATASET_SIZE = 1000

files = ['train/' + f for f in os.listdir('train')]
np.random.shuffle(files)
files = files[:DATASET_SIZE]

cats = [decode_jpg(f) for f in files if 'cat' in f]
dogs = [decode_jpg(f) for f in files if 'dog' in f]

show(random.choice(cats))
print("Aww, a kitty!\n")

show(random.choice(dogs))
print("Who's a good dog!\n")

print("Loading VGG16 pretrained on Imagenet")
vgg = applications.vgg16.VGG16()

CAT_CLASSES = range(281, 294)
DOG_CLASSES = range(151, 275)

for _ in range(1000):
    image = random.choice(cats + dogs)
    image = np.expand_dims(image, axis=0)
    pred = vgg.predict(image)
    is_cat = np.argmax(pred) in CAT_CLASSES
    is_dog = np.argmax(pred) in DOG_CLASSES

    show(image)
    if is_cat:
        print("I think this is a cat with probability {:.3f}".format(pred.max()))
    elif is_dog:
        print("I think this is a dog with probability {:.3f}".format(pred.max()))
    else:
        print("I don't know what this is")
    print('\n')
    time.sleep(.5)
