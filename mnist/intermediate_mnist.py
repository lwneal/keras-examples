import numpy as np
from keras import layers, models, datasets

from imutil import show

print("Loading MNIST Data")
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()


# This is a super-simple model: it's just a linear classifier!
input_shape = (28, 28)
model = models.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
model.summary()

while True:
    i = np.random.randint(len(x_test))
    x = x_test[i:i+9]
    show(x)
    y = model.predict(x)
    y = np.argmax(y, axis=-1)
    print("I think these numbers are: {}".format(y))

    model.fit(x_train, y_train, epochs=1)
