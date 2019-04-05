import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
import tensorflow as tf
from random import randint

length =5
nr_features = 10
out_index = 2

def generate_sequence(length,nr_features):
    return [randint(0,nr_features-1) for _ in range(length)]

def one_hot_enconde(sequence,nr_features):
    encoded = list()
    for value in sequence:
        one_hot_enconded = np.zeros(nr_features)
        one_hot_enconded[value] = 1
        encoded.append(one_hot_enconded)
    return np.array(encoded)

def one_hot_decode(enconded_seq):
    return [np.argmax(value) for value in enconded_seq]

def generate_sample(length,nr_features,out_index):
    sequence = generate_sequence(length,nr_features)
    enconded = one_hot_enconde(sequence,nr_features)
    X = enconded.reshape((1, length,nr_features))
    y = enconded[out_index].reshape(1,nr_features)
    return X,y

#define model

model = Sequential()
model.add(LSTM(15, input_shape=(length,nr_features)))
model.add(Dense(nr_features,activation="softmax"))

# compile da model

model.compile(loss= tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

# fit the model

for i in range(10000):
    X,y = generate_sample(length,nr_features,out_index)
    model.fit(X,y,shuffle=False,epochs=1,verbose=2)

# evaluate

correct = 0
for i in range(100):
    X,y = generate_sample(length,nr_features,out_index)
    predict_y = model.predict(X)
    print('Sequence %s' % [one_hot_decode(x) for x in X])
    print('Expected: %s' % one_hot_decode(y))
    print('Predicted: %s' % one_hot_decode(predict_y))
    if (one_hot_decode(predict_y) == one_hot_decode(y)):
        correct += 1
print('Accuracy: %f' % ((correct/100)*100.0))


