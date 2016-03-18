import sys,os
from PIL import Image, ImageFilter
import numpy as np
import h5py
import tables
import gc
import temporal_stream_data as tsd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization

def getTrainData():
	train=tsd.stackOF()
	for X_train,Y_train in train:
		X_train /= 255
		yield (X_train,Y_train)

def getTrainData():
	pass

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations


def CNN():
	input_frames=10
	batch_size = 32
	nb_classes = 20
	nb_epoch = 200
	img_rows, img_cols = 224,224
	img_channels = 2*input_frames

	print 'X_sample: '+str(X_sample.shape)
	print 'X_test: '+str(X_test.shape)
	print 'Y_test: '+str(Y_test.shape)


	print 'Preparing architecture...'

	model = Sequential()

	model.add(Convolution2D(96, 7, 7, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, 5, 5, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(512, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(512, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Convolution2D(512, 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(2048))
	fc_output=Activation('relu')
	model.add(fc_output)
	model.add(Dropout(0.5))
	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))
	softmax_output=Activation('softmax')
	model.add(softmax_output)



	print 'Starting with training...'
	gc.collect()
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss={'output1':'categorical_crossentropy'}, optimizer=sgd)

	print("Using real time data augmentation")

	datagen = ImageDataGenerator(
		featurewise_center=True,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=True,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=True)  # randomly flip images

	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen.fit(X_sample)

	for e in range(nb_epoch):
		print('-'*40)
		print('Epoch', e)
		print('-'*40)
		print("Training...")
		# batch train with realtime data augmentation
		progbar = generic_utils.Progbar(X_train.shape[0])
		for X_train, Y_train in getTrainData():
			for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
				loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)
				progbar.add(X_batch.shape[0], values=[("train loss", loss[0]),("train accuracy", loss[1])])
				fc_output
				softmax_output

		print('Saving layer representation and saving weights...')

		with h5py.File('fc_output.h5', 'w') as hf:
			hf.create_dataset('fc_output', data=fc_output)

		with h5py.File('softmax_output.h5', 'w') as hf:
			hf.create_dataset('softmax_output', data=softmax_output)

		model.save_weights('temporal_stream_model.h5')

		print("Testing...")
		# test time!
		progbar = generic_utils.Progbar(X_test.shape[0])
		for X_test, Y_test in getTestData():
			for X_batch, Y_batch in datagen.flow(X_test, Y_test, batch_size=batch_size):
				score = model.test_on_batch(X_batch, Y_batch, accuracy=True)
				progbar.add(X_batch.shape[0], values=[("test loss", score[0]),("test accuracy", score[1])])


if __name__ == "__main__":
	CNN()