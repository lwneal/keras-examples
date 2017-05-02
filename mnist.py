import numpy as np
from keras import layers, models, datasets

import imutil

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

imutil.show(x_train[123])
