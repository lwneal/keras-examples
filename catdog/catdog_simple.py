import os
from imutil import *

from keras import models, layers, applications
import numpy as np

files = ['train/' + f for f in os.listdir('train')]
np.random.shuffle(files)
files = files[:1000]

cats = [decode_jpg(f) for f in files if 'cat' in f]
dogs = [decode_jpg(f) for f in files if 'dog' in f]

vgg = applications.vgg16.VGG16()

