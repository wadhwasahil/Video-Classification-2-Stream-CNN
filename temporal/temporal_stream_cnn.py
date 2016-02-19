import sys,os
from PIL import Image, ImageFilter
import numpy as np
import h5py
import gc

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization

def getData():
	return (X_train,Y_train,X_test,Y_test)

def CNN(X_train,Y_train,X_test,Y_test):

	input_frames=10
	batch_size = 32
	nb_classes = 11
	nb_epoch = 200
	img_rows, img_cols = 256,256
	img_channels = 2*input_frames

	print 'Readying vectors...'
	gc.collect()

	X_train = X_train.astype("float16",copy=False)
	X_test = X_test.astype("float16",copy=False)
	# X_train /= 255
	# X_test /= 255

	print X_train.shape
	print X_test.shape
	print Y_train.shape
	print Y_test.shape

	print 'Preparing architecture...'


	graph = Graph()

	graph.add_input(name='input1',input_shape=(img_channels, img_rows, img_cols))

	graph.add_node(Convolution2D(96, 7, 7, border_mode='same'),name='conv1',input='input1')
	graph.add_node(BatchNormalization(),name='norm1',input='conv1')
	graph.add_node(Activation('relu'),name='act1',input='norm1')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool1',input='act1')

	graph.add_node(Convolution2D(256, 5, 5, border_mode='same'),name='conv2',input='pool1')
	graph.add_node(BatchNormalization(),name='norm2',input='conv2')
	graph.add_node(Activation('relu'),name='act2',input='norm2')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool2',input='act2')

	graph.add_node(Convolution2D(512, 3, 3, border_mode='same'),name='conv3',input='pool2')
	graph.add_node(BatchNormalization(),name='norm3',input='conv3')
	graph.add_node(Activation('relu'),name='act3',input='norm3')

	graph.add_node(Convolution2D(512, 3, 3, border_mode='same'),name='conv4',input='act3')
	graph.add_node(BatchNormalization(),name='norm4',input='conv4')
	graph.add_node(Activation('relu'),name='act4',input='norm4')

	graph.add_node(Convolution2D(512, 3, 3, border_mode='same'),name='conv5',input='act4')
	graph.add_node(BatchNormalization(),name='norm5',input='conv5')
	graph.add_node(Activation('relu'),name='act5',input='norm5')
	graph.add_node(MaxPooling2D(pool_size=(2, 2)),name='pool3',input='act5')

	graph.add_node(Flatten(),name='flat',input='pool3')
	graph.add_node(Dense(4096),name='fc1',input='flat')
	graph.add_node(Activation('relu'),name='act6',input='fc1')
	graph.add_node(Dropout(0.5),name='drop1',input='act6')
	graph.add_node(Dense(2048),name='fc2',input='drop1')
	graph.add_node(Activation('relu'),name='act7',input='fc2')
	graph.add_node(Dropout(0.5),name='drop2',input='act7')

	graph.add_node(Dense(nb_classes),name='out',input='drop2')
	graph.add_node(Activation('softmax'),name='soft',input='out')
	graph.add_output(name='output1',input='soft')

	print 'Starting with training...'
	gc.collect()
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	graph.compile(loss={'output1':'categorical_crossentropy'}, optimizer=sgd)
	history = graph.fit({'input1':X_train, 'output1':Y_train},nb_epoch=nb_epoch,batch_size=batch_size,verbose=1)
	score = graph.evaluate({'input1':X_test, 'output1':Y1_test}, batch_size=batch_size)
	print score

data=getData()
CNN(data[0],data[1],data[2],data[3])