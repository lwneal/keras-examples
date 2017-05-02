import numpy as np
from keras import layers, models, datasets

import imutil

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

print("MNIST Training Data:")
for i in range(3):
    imutil.show(x_train[i])
    print("Label: {}".format(y_train[i]))
