from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from keras import datasets, layers, models, Model

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot
import h5py
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(2022)
VALIDATION_SIZE = 5000
# Pour telécharger la database
from matplotlib import pyplot
from scipy import misc


def prepare_MNIST_data(use_data_augmentation=True):
	(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

#data augmentation
	
	datagen = ImageDataGenerator( rotation_range=90,
                 	width_shift_range=0.1, height_shift_range=0.1,
                 	horizontal_flip=True)
	datagen.fit(train_data)


 
 
	(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
	train_data = train_data.astype('float32')
	test_data = test_data.astype('float32')
 
#z-score
	mean = np.mean(train_data,axis=(0,1,2,3))
	std = np.std(train_data,axis=(0,1,2,3))
	train_data = (train_data-mean)/(std+1e-7)
	test_data = (test_data-mean)/(std+1e-7)
 
	num_classes = 10
	train_labels = np_utils.to_categorical(train_labels,num_classes)
	test_labels = np_utils.to_categorical(test_labels,num_classes)
	validation_data = train_data[:VALIDATION_SIZE, :]
	validation_labels = train_labels[:VALIDATION_SIZE,:]
	train_data = train_data[VALIDATION_SIZE:, :]
	train_labels = train_labels[VALIDATION_SIZE:,:] 
	return train_data, train_labels, validation_data, validation_labels, test_data, test_labels
