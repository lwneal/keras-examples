import numpy as np
from keras import layers, models, datasets

from imutil import show

print("Loading MNIST Data")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

input_shape = (28, 28, 1)

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

while True:
    model.fit(x_train, y_train, epochs=1)
    i = np.random.randint(len(x_test))
    x = x_test[i:i+4]
    show(x)
    pred = model.predict(x)
    print("I think these numbers are {}".format(np.argmax(pred, axis=-1)))
