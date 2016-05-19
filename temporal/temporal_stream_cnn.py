import numpy as np
import h5py
import gc
import temporal_stream_data as tsd
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from six.moves import range
from keras.layers.normalization import BatchNormalization


def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def get_activations(model, layer, X_batch):
	get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
	activations = get_activations(X_batch)
	return activations

def getTrainData(chunk,nb_classes,img_rows,img_cols):
	X_train,Y_train=tsd.stackOF(chunk,img_rows,img_cols)
	if (X_train!=None and Y_train!=None):
		X_train/=255
		# X_train=X_train-np.average(X_train)
		Y_train=np_utils.to_categorical(Y_train,nb_classes)
	return (X_train,Y_train)


def CNN():
	input_frames=10
	batch_size=8
	nb_classes = 20
	nb_epoch = 200
	img_rows, img_cols = 150,150
	img_channels = 2*input_frames
	chunk_size=64
	print 'Loading dictionary...'

	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)


	print 'Preparing architecture...'

	model = Sequential()

	model.add(Convolution2D(48, 7, 7, border_mode='same',input_shape=(img_channels, img_rows, img_cols)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(96, 5, 5, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(256, 3, 3, border_mode='same'))
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
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.7))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.8))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	# model.load_weights('temporal_stream_model.h5')



	print 'Compiling model...'
	gc.collect()
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True,clipnorm=0.1)
	model.compile(loss='categorical_crossentropy',optimizer=sgd)

	for e in range(nb_epoch):
		print('-'*40)
		print('Epoch', e)
		print('-'*40)
		instance_count=0

		flag=0
		keys=temporal_train_data.keys()
		random.shuffle(keys)

		for chunk in chunks(keys,chunk_size):
			if flag<1:
				print("Preparing testing data...")
				X_test,Y_test=getTrainData(chunk,nb_classes,img_rows,img_cols)
				flag+=1
				continue
			print instance_count
			instance_count+=chunk_size
			X_batch,Y_batch=getTrainData(chunk,nb_classes,img_rows,img_cols)
			if (X_batch!=None and Y_batch!=None):
				loss = model.fit(X_batch, Y_batch, verbose=1, batch_size=batch_size, nb_epoch=1, show_accuracy=True)	
				if instance_count%256==0:
					loss = model.evaluate(X_test,Y_test,batch_size=batch_size,verbose=1)
					preds = model.predict(X_test)
					print (preds)
					print ('-'*40)
					print (Y_test)
					comparisons=[]
					maximum=np.argmax(Y_test,axis=1)
					for i,j in enumerate(maximum):
						comparisons.append(preds[i][j])
					with open('compare.txt','a') as f1:
						f1.write(str(comparisons))
						f1.write('\n\n')
					with open('loss.txt','a') as f1:
						f1.write(str(loss))
						f1.write('\n')
					model.save_weights('temporal_stream_model.h5',overwrite=True)


if __name__ == "__main__":
	CNN()
