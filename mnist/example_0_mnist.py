import numpy as np
from keras import layers, models, datasets

from imutil import show

# Keras has some useful functions to download data automatically
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# We always need to specify the input shape
input_shape = (28, 28)

# This is a super-simple model: it's just a linear classifier!
model = models.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(10))

# This is the softmax function: it normalizes our output to a magnitude of 1
model.add(layers.Activation('softmax'))

# We choose the loss function for convenience, so we can input integers to the model
model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam', 
        metrics=['accuracy'])  # Don't mind the optimizer and metrics just yet
model.summary()

while True:
    # Run stochastic gradient descent for 1 epoch
    model.fit(x_train, y_train, epochs=1)

    # Choose a random image from the test set (not the training set!)
    i = np.random.randint(len(x_test))
    x = x_test[i:i+4]

    # Display it on the screen
    show(x)

    # Also display the model's prediction
    y = model.predict(x)
    y = np.argmax(y, axis=-1)
    print("I think these numbers are {}".format(y))
