import numpy as np
from keras import layers, models, datasets
from imutil import show

# The images in CIFAR-10 are real photographs
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print("Loading CIFAR-10 Data")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Our image now has 3 channels -  Red, Green, and Blue
input_shape = (32, 32, 3)

# Two convolutions with max pooling into a dense layer
# ... but now with Batch Normalization!
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
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Now it works, because of Batch Normalization!
while True:
    # Select a few random images
    i = np.random.randint(len(x_test))
    x = x_test[i:i+4]
    show(x)

    # Print predictions of the model on these images
    y = model.predict(x)
    y = np.argmax(y, axis=-1)
    y = [classes[c] for c in y]
    print("I think these images are {}".format(y))

    model.fit(x_train, y_train, epochs=1)
