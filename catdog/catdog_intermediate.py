import time
import os
import random
from imutil import show, decode_jpg

from keras import models, layers, applications
import numpy as np

def load_data(DATASET_SIZE=1000):
    files = ['train/' + f for f in os.listdir('train')]
    np.random.shuffle(files)
    files = files[:DATASET_SIZE]

    # Cats will be class 0, and dogs will be class 1
    cats = [(decode_jpg(f), 0) for f in files if 'cat' in f]
    dogs = [(decode_jpg(f), 1) for f in files if 'dog' in f]
    images = cats + dogs
    np.random.shuffle(images)
    x, y = zip(*images)
    return np.array(x), np.array(y)

def demo(model):
    image, label = load_data(1)
    show(image)
    pred = model.predict(image)
    if np.argmax(pred) == 0:
        print("This is a cat with probability {:.2f}".format(pred.max()))
    else:
        print("This is a dog with probability {:.2f}".format(pred.max()))
    print('\n')


print("Loading VGG16 pretrained on Imagenet...")
vgg = applications.vgg16.VGG16()
print("VGG16 loaded with {} layers".format(len(vgg.layers)))

model = models.Sequential()
# Add all the layers except the last one
for layer in vgg.layers[:-1]:
    layer.trainable = False
    model.add(layer)
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])

for _ in range(10):
    print("Loading data...")
    x_train, y_train = load_data(1000)
    x_test, y_test = load_data(100)
    print("Loaded data")
    for _ in range(4):
        demo(model)
        time.sleep(1)
    model.fit(x_train, y_train, epochs=1)
    metrics = model.evaluate(x_test, y_test)
    print("Test set accuracy is: {}".format(metrics[-1]))

