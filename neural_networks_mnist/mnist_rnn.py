#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:51:27 2018

@author: kseniyagrasdal
"""

import matplotlib.pyplot as plt
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
import keras
import sys
from keras.layers import Activation, Dropout, Dense #, MaxPooling2D
from keras.models import Sequential 
from keras import optimizers
from keras.layers.recurrent import LSTM
from numpy.random import seed 
from tensorflow import set_random_seed


# Courtesy https://github.com/sorki for providing a data parser
# Get package from PyPi with pip install python-mnist
from mnist import MNIST 
import numpy as np
from keras.utils import to_categorical

mndata = MNIST('./dataset/')

images, labels = mndata.load_training()
X_train_ar, y_train_ar = np.array(images)[0:1000], np.array(labels)[0:1000]

images, labels = mndata.load_testing()
X_test_ar, y_test_ar = np.array(images)[0:200], np.array(labels)[0:200]


# Part II - RNN (LSTM)
    
# Preprocess data, reshape to 1-channel images
X_train = X_train_ar.reshape(X_train_ar.shape[0], 28, 28).astype('float32') / 255
X_test = X_test_ar.reshape(X_test_ar.shape[0], 28, 28).astype('float32') / 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# Convert a class vector to binary class matrix
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train_ar, num_classes = 10)
y_test = np_utils.to_categorical(y_test_ar, num_classes = 10)

# Build simple RNN (LSTM)

def LSTM_model(hidden_units = 250, units =10, input_shape = (28, 28)):
    model = Sequential()
    model.add(LSTM(hidden_units, return_sequences = True, input_shape = input_shape))
 #   model.add(Dropout(0.2))
    
    model.add(LSTM(hidden_units, return_sequences = True))
 #   model.add(Dropout(0.2))
    
    model.add(LSTM(hidden_units, return_sequences = True))
 #   model.add(Dropout(0.2))
    
    model.add(LSTM(hidden_units, return_sequences = False))
 #   model.add(Dropout(0.2))
    
    model.add(Dense(units = units))
    model.add(Activation('softmax', name='output'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adam(lr=0.0005),
                  metrics={'output': 'accuracy'})
    return(model)
    
model_rnn = LSTM_model()
model_rnn.summary()

# Train RNN (LSTM) model
import time
start = time.time()
seed(0)
set_random_seed(0)
hist_rnn = model_rnn.fit(X_train, y_train, batch_size = 64, epochs = 50,
          validation_data=[X_test, y_test], verbose = 1)
print('It took {0:0.1f} seconds'.format(time.time() - start))

# Evaluate performance of RNN (LSTM) model   
scores = model_rnn.evaluate(X_test, y_test, verbose = 1)
print('Test cross-entropy loss:', scores[0])
print('Test accuracy:', scores[1])

# Visualize trajectory of training
plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
for label in ['val_acc', 'acc']:
    plt.plot(hist_rnn.history[label], label=label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy, model1:{:6.5f}'.format(hist_rnn.history['val_acc'][-1]))
plt.legend()

plt.subplot(1,2,2)
for label in ['val_loss', 'loss']:
    plt.plot(hist_rnn.history[label], label=label)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Validation Loss, model1:{:6.5f}'.format(hist_rnn.history['val_loss'][-1]))

plt.legend()
    
# Inspect output of RNN (LSTM) model
predicted_classes = model_rnn.predict_classes(X_test)
correct_indices   = np.nonzero(predicted_classes == y_test.argmax(axis=-1))[0]
incorrect_indices = np.nonzero(predicted_classes != y_test.argmax(axis=-1))[0]

plt.figure(1, figsize=(7,7))
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])
    
plt.figure(2, figsize=(7,7))
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect].argmax(axis=-1)))
    plt.xticks([])
    plt.yticks([])