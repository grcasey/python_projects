# MNIST Dataset - СNN

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import sys

print('python{})'.format(sys.version))
print('keras version {}'.format(keras.__version__))
print('tensorflow version {}'.format(tf.__version__))

# Check what device is TensorFlow running at
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# #Set to use GPU - for GPU enabled devices
###############################################################################
# #Creates a graph.
#with tf.device('/device:GPU:1'):
 # a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
 # b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
 # c = tf.matmul(a, b)
 
# #Creates a session with allow_soft_placement and log_device_placement set
# #to True.
#sess = tf.Session(config=tf.ConfigProto(
#       log_device_placement=True,
#       inter_op_parallelism_threads=1,
#       intra_op_parallelism_threads=1))
# #Runs the op.
#print(sess.run(c))

#config = tf.ConfigProto()
# #Allocate up to 95% of GPU memory to each process
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
# #Specify GPU ids to determine 'visible' to 'virtual' mapping
#config.gpu_options.visible_device_list = '3'

#set_session(tf.Session(config=config))
###############################################################################



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

# Part I - CNN

# Preprocess data, reshape to 1-channel images, courtesy https://bit.ly/2NNnzZF
# Also, original data is uint8 (0-255). Normalize by scaling to range [0,1] (dividing by 255)

X_train = X_train_ar.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test_ar.reshape(-1, 28, 28, 1).astype('float32') / 255

y_train = to_categorical(y_train_ar.astype('float32'))
y_test = to_categorical(y_test_ar.astype('float32'))
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Visualize 10 first digits of the data
plt.figure(1, figsize=(14,3))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

# Build CNN, inspired by (Sabour, Frosst & Hinton, 2017)
from keras.layers import Conv2D, Flatten, Dropout, Activation, Dense #, MaxPooling2D
from keras.models import Sequential 
from keras import optimizers
from numpy.random import seed 
from tensorflow import set_random_seed

def CNN_model(n_class=10,
              input_shape = (28, 28, 1)):
    
    model = Sequential()
    model.add(Conv2D(256,(5, 5), strides=1,
                     activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(Conv2D(256,(5, 5), strides=1,
                     activation='relu', padding='same'))
    model.add(Conv2D(128,(5, 5), strides=1,
                     activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(328, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class))
    model.add(Activation('softmax', name='output'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0005),
                  metrics={'output': 'accuracy'})
    return(model)
    
model_cnn = CNN_model()
model_cnn.summary()   #35M parameters

# Train CNN model
import time
start = time.time()
seed(1)
set_random_seed(1)

hist = model_cnn.fit(X_train, y_train,
                        batch_size=64, epochs=50,
                        validation_data=[X_test, y_test], verbose=1)

print('It took {0:0.1f} seconds'.format(time.time() - start))
    
# Evaluate performance of CNN model         
score = model_cnn.evaluate(X_test, y_test, verbose=0)
print('Test cross-entropy loss: %0.5f' % score[0])
print('Test accuracy: %0.2f' % score[1])

# Visualize trajectory of training
plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
for label in ['val_acc', 'acc']:
    plt.plot(hist.history[label], label=label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy, model1:{:6.5f}'.format(hist.history['val_acc'][-1]))
plt.legend()

plt.subplot(1,2,2)
for label in ['val_loss', 'loss']:
    plt.plot(hist.history[label], label=label)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Validation Loss, model1:{:6.5f}'.format(hist.history['val_loss'][-1]))

plt.legend()

# Inspect first 9 digits of an output based on correct and incorrect predictions
# Visualization is courtesy of https://bit.ly/2yx9j2I
predicted_classes = model_cnn.predict_classes(X_test)
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
