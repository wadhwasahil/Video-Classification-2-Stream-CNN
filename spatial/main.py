from keras.models import Sequential, Graph
import h5py
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import cv2, numpy as np

def VGG_16(weights_path=None):
    model1 = Graph()
    model1.add_input('input', input_shape = (3,224,224))
    model1.add_node(ZeroPadding2D((1,1), name = "input1", input = "input")
    model1.add_node(Convolution2D(64, 3, 3, activation='relu'), name = "conv1", input = "input1")
    model1.add_node(ZeroPadding2D((1,1)), name = "input2", input = "conv1")
    model1.add_node(Convolution2D(64, 3, 3, activation='relu'), name = "conv2", input = "input2")
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool1", input = "conv2")
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input3", input = "pool1")
    model1.add_node(Convolution2D(128, 3, 3, activation='relu'), name = "conv3", input = "input3")
    model1.add_node(ZeroPadding2D((1,1)), name = "input4", input = "conv3")
    model1.add_node(Convolution2D(128, 3, 3, activation='relu'), name = "conv4", input = "input4")
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool2", input = "conv4")
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input5", input = "pool2")
    model1.add_node(Convolution2D(256, 3, 3, activation='relu'), name = "conv5", input = "input5")
    model1.add_node(ZeroPadding2D((1,1)), name = "input6", input = "conv5")
    model1.add_node(Convolution2D(256, 3, 3, activation='relu'), name = "conv6", input = "input6")
    model1.add_node(ZeroPadding2D((1,1)), name = "input7", input = "conv6")
    model1.add_node(Convolution2D(256, 3, 3, activation='relu'), name = "conv7", input = "input7")
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool3", input = "conv7")
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input8", input = "pool3")
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv8", input = "input8")
    model1.add_node(ZeroPadding2D((1,1)), name = "input9", input = "conv8")
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv9", input = "input9")
    model1.add_node(ZeroPadding2D((1,1)), name = "input10", input = "conv9")
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv10", input = "input10")
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool4", input = "conv10")
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input11", input = "pool4")
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv11", input = "input11")
    model1.add_node(ZeroPadding2D((1,1)), name = "input12", input = "conv11")
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv12", input = "input12")
    model1.add_node(ZeroPadding2D((1,1)), name = "input13", input = "conv12")
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv13", input = "input13")
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool5", input = "conv13")
 
    model1.add_node(Flatten(), name = "flat1", input = "pool5")
    model1.add_node(Dense(4096, activation='relu'), name = "fc1", input = "flat1")
    model1.add_node(Dropout(0.5), name = "drop1", input = "fc1")
    model1.add_node(Dense(4096, activation='relu'), name = "fc2", input = "drop1")
    model1.add_node(Dropout(0.5), name = "drop2", input = "fc2")
    model1.add_node(Dense(1000, activation='softmax'), name = "fc3", input = "drop2")
    model1.add_output(name = "out", input = "fc3")
    return model1   

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model1 = VGG_16()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(optimizer=sgd, loss={"out" : 'categorical_crossentropy'})
    # out = model.predict(im)
    # c = np.argmax(out)
    # print c