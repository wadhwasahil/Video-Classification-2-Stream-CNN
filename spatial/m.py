from keras.models import Sequential,Graph
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

m = Sequential()

m.load_weights("vgg16_weights.h5")
print m.optimizers.lr