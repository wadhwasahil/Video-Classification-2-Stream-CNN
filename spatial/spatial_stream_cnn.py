from read_data import get_train_data, get_test_data, get_sample_data
import random
import cv2, numpy as np
import pickle
import h5py

from keras.models import Sequential, Graph
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def getSampleData(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_sample_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        X_train/=255
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
    return (X_train,Y_train)

def getTestData(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_test_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        X_train/=255
        # X_train=X_train-np.average(X_train)
        Y_train=np_utils.to_categorical(Y_train,nb_classes)
    return (X_train,Y_train)

def getTrainData(chunk,nb_classes,img_rows,img_cols):
    X_train,Y_train = get_train_data(chunk,img_rows,img_cols)
    if (X_train!=None and Y_train!=None):
        X_train/=255
        # X_train=X_train-np.average(X_train)
        # Y_train=np_utils.to_categorical(Y_train,nb_classes)
    return (X_train,Y_train)

def test(model, nb_epoch, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size):
    keys=spatial_test_data.keys()
    random.shuffle(keys)
    X_test,Y_test = getTestData(spatial_test_data.keys()[:500],nb_classes,img_rows,img_cols)
    return (X_test, Y_test)


def train(model, nb_epoch, spatial_train_data, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size):

    datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images

    X_sample,Y_sample=getSampleData(spatial_train_data.keys()[:chunk_size],nb_classes,img_rows,img_cols)
    datagen.fit(X_sample)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        instance_count=0

        X_test,Y_test = test(model, nb_epoch, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size)

        keys=spatial_train_data.keys()
        random.shuffle(keys)
        for chunk in chunks(keys,chunk_size):
            X_chunk,Y_chunk=getTrainData(chunk,nb_classes,img_rows,img_cols)
            if (X_chunk!=None and Y_chunk!=None):
                #for X_batch, Y_batch in datagen.flow(X_chunk, Y_chunk, batch_size=chunk_size):
                loss = model.fit(X_chunk, Y_chunk, verbose=1, batch_size=batch_size, nb_epoch=1, show_accuracy=True, validation_data=(X_test,Y_test))
                instance_count+=chunk_size
                print instance_count
                if instance_count%256==0:
                    model.save_weights('spatial_stream_model.h5',overwrite=True)

def VGG_16(img_rows,img_cols,weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(20, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
	return model  

if __name__ == "__main__":

    nb_epoch = 50
    batch_size = 2
    nb_classes = 20
    chunk_size = 32
    img_rows = 224
    img_cols = 224
    model =[]
    print 'Making model...'
    model = VGG_16(img_rows,img_cols,'vgg16_weights.h5')

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    print 'Compiling model...'
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print 'Loading train dictionary...'
    with open('../dataset/spatial_train_data_new.pickle','rb') as f1:
        spatial_train_data=pickle.load(f1)

    print 'Loading test dictionary...'
    with open('../dataset/spatial_test_data.pickle','rb') as f1:
        spatial_test_data=pickle.load(f1)

    print 'Training model...'
    train(model, nb_epoch, spatial_train_data, spatial_test_data, chunk_size, nb_classes, img_rows, img_cols, batch_size)