import numpy as np
from keras import layers, models, datasets

from imutil import show

# The images in CIFAR-10 are real photographs (but at a low resolution)
# There are 10 classes, labeled below
def class_name(number):
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][number]


print("Loading CIFAR-10 Data")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Our image now has 3 channels -  Red, Green, and Blue
input_shape = (32, 32, 3)

# We use the same network as before!
model = models.Sequential()
model.add(layers.Conv2D(16, (3,3), input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, (3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(32))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# This is also the same
while True:
    model.fit(x_train, y_train, epochs=1)
    i = np.random.randint(len(x_test))
    x = x_test[i:i+4]
    show(x)
    y = model.predict(x)
    y = np.argmax(y, axis=-1)
    y = map(class_name, y)
    print("I think these images are {}".format(y))
