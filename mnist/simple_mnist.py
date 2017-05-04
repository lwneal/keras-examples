import numpy as np
from keras import layers, models, datasets

# Keras provides this useful function for loading MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# This is a super-simple model: a linear classifier with a softmax
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# This performs gradient descent automatically
model.fit(x_train, y_train, epochs=1)

# The model is now trained and can predict on new data
y = model.predict(x_test[123:133])
y = np.argmax(y, axis=-1)
print("Ground Truth: {}".format(y_test[123:133]))
print("Prediction:  {}".format(y))
