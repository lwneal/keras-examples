'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 30  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Now we demonstrate...
import numpy as np
from keras import datasets, models, layers

word_to_idx = datasets.imdb.get_word_index()
# Fix bug in Keras 2.0.2
for word in word_to_idx:
    word_to_idx[word] += 3

idx_to_word = {v:k for (k,v) in word_to_idx.items()}
def to_str(index_list):
    # Hack for https://github.com/fchollet/keras/issues/5912
    index_list = [max(1,i-3) for i in index_list]
    word_list = map(word_to_idx.get, index_list)
    return ' '.join(word_list)

while True:
    print('Training the model...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    print("\n")
    print("\n")
    words = raw_input("Input a sentence: ").lower().split()
    indices = map(word_to_idx.get, words)
    indices = [i if i is not None else 0 for i in indices]

    print("Converted sentence into: {}".format(indices))
    x = np.expand_dims(indices, axis=0)
    score = model.predict(x)[0]
    print("Score for this sentence is: {}".format(score))
    print('\n')
