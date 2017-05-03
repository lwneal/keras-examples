import numpy as np
from keras import layers, models, datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# This is a super-simple model: it's just a linear classifier!
input_shape = (28, 28)
model = models.Sequential()
model.add(layers.Flatten(input_shape=input_shape))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

x = x_test[100:110]
y_true = y_test[100:110]
print("Ground Truth: {}".format(y_true))

y = model.predict(x)
y = np.argmax(y, axis=-1)
print("Prediction:  {}".format(y))
