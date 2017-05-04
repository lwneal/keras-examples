import numpy as np
from keras import layers, models, datasets

from imutil import show

print("Loading MNIST Data")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# We add the extra dimension here. This is just for compatibility.
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Now our input shape has an extra dimension
input_shape = (28, 28, 1)

# Look at this model- it's larger!
model = models.Sequential()

# We're using convolutional layers
model.add(layers.Conv2D(16, (3,3), input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, (3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())

# Now we have multiple layers of nonlinearity
model.add(layers.Dense(32))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

# Same loss function and gradient descent as before, but with more parameters
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

while True:
    # This is all the same as before
    model.fit(x_train, y_train, epochs=1)
    i = np.random.randint(len(x_test))
    x = x_test[i:i+4]
    show(x)
    y = model.predict(x)
    y = np.argmax(y, axis=-1)
    print("I think these numbers are {}".format(y))
