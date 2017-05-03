import numpy as np
from keras import layers, models, datasets

from imutil import show

# Same classes
def class_name(number):
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][number]


print("Loading CIFAR-10 Data")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

input_shape = (32, 32, 3)  # Same input

model = models.Sequential()
model.add(layers.Conv2D(64, (3,3), input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, (3,3)))  # Bigger convolutional layers!
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dense(128))        # More fully-connected layers!
model.add(layers.BatchNormalization())  # As always, we use lots of BatchNormalization, can't get enough of that
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

while True:
    # Specify epochs=1 to train only one epoch (default is 10)
    model.fit(x_train, y_train, epochs=1)

    # Select a random 4 images from x_test and show them
    i = np.random.randint(len(x_test))
    x = x_test[i:i+4]
    show(x)

    # Convert the network's softmax output to integers using argmax
    y = model.predict(x)
    y = np.argmax(y, axis=-1)
    y = map(class_name, y)
    print("I think these images are {}".format(y))

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test Accuracy: {}".format(acc))
