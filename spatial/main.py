from keras.models import Sequential, Graph
import h5py
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import cv2, numpy as np
from numpy import array

def VGG_16(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
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
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    # print model.layers[36].get_weights()    
    model1 = Graph()
    model1.add_input(name = "input", input_shape=(3,224,224))
    model1.add_node(ZeroPadding2D((1,1),input_shape=(3,  224,224)), name = "input1", input = "input")
    model1.nodes['input1'].set_weights(model.layers[0].get_weights())
    model1.add_node(Convolution2D(64, 3, 3, activation='relu'), name = "conv1", input = "input1")
    model1.nodes['conv1'].set_weights(model.layers[1].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input2", input = "conv1")
    model1.nodes['input2'].set_weights(model.layers[2].get_weights())
    model1.add_node(Convolution2D(64, 3, 3, activation='relu'), name = "conv2", input = "input2")
    model1.nodes['conv2'].set_weights(model.layers[3].get_weights())
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool1", input = "conv2")
    model1.nodes['pool1'].set_weights(model.layers[4].get_weights())
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input3", input = "pool1")
    model1.nodes['input3'].set_weights(model.layers[5].get_weights())
    model1.add_node(Convolution2D(128, 3, 3, activation='relu'), name = "conv3", input = "input3")
    model1.nodes['conv3'].set_weights(model.layers[6].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input4", input = "conv3")
    model1.nodes['input4'].set_weights(model.layers[7].get_weights())
    model1.add_node(Convolution2D(128, 3, 3, activation='relu'), name = "conv4", input = "input4")
    model1.nodes['conv4'].set_weights(model.layers[8].get_weights())
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool2", input = "conv4")
    model1.nodes['pool2'].set_weights(model.layers[9].get_weights())
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input5", input = "pool2")
    model1.nodes['input5'].set_weights(model.layers[10].get_weights())
    model1.add_node(Convolution2D(256, 3, 3, activation='relu'), name = "conv5", input = "input5")
    model1.nodes['conv5'].set_weights(model.layers[11].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input6", input = "conv5")
    model1.nodes['input6'].set_weights(model.layers[12].get_weights())
    model1.add_node(Convolution2D(256, 3, 3, activation='relu'), name = "conv6", input = "input6")
    model1.nodes['conv6'].set_weights(model.layers[13].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input7", input = "conv6")
    model1.nodes['input7'].set_weights(model.layers[14].get_weights())
    model1.add_node(Convolution2D(256, 3, 3, activation='relu'), name = "conv7", input = "input7")
    model1.nodes['conv7'].set_weights(model.layers[15].get_weights())
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool3", input = "conv7")
    model1.nodes['pool3'].set_weights(model.layers[16].get_weights())
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input8", input = "pool3")
    model1.nodes['input8'].set_weights(model.layers[17].get_weights())
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv8", input = "input8")
    model1.nodes['conv8'].set_weights(model.layers[18].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input9", input = "conv8")
    model1.nodes['input9'].set_weights(model.layers[19].get_weights())
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv9", input = "input9")
    model1.nodes['conv9'].set_weights(model.layers[20].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input10", input = "conv9")
    model1.nodes['input10'].set_weights(model.layers[21].get_weights())
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv10", input = "input10")
    model1.nodes['conv10'].set_weights(model.layers[22].get_weights())
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool4", input = "conv10")
    model1.nodes['pool4'].set_weights(model.layers[23].get_weights())
 
    model1.add_node(ZeroPadding2D((1,1)), name = "input11", input = "pool4")
    model1.nodes['input11'].set_weights(model.layers[24].get_weights())
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv11", input = "input11")
    model1.nodes['conv11'].set_weights(model.layers[25].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input12", input = "conv11")
    model1.nodes['input12'].set_weights(model.layers[26].get_weights())
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv12", input = "input12")
    model1.nodes['conv12'].set_weights(model.layers[27].get_weights())
    model1.add_node(ZeroPadding2D((1,1)), name = "input13", input = "conv12")
    model1.nodes['input13'].set_weights(model.layers[28].get_weights())
    model1.add_node(Convolution2D(512, 3, 3, activation='relu'), name = "conv13", input = "input13")
    model1.nodes['conv13'].set_weights(model.layers[29].get_weights())
    model1.add_node(MaxPooling2D((2,2), strides=(2,2)), name = "pool5", input = "conv13")
    model1.nodes['pool5'].set_weights(model.layers[30].get_weights())
 
    model1.add_node(Flatten(), name = "flat1", input = "pool5")
    model1.nodes['flat1'].set_weights(model.layers[31].get_weights())
    model1.add_node(Dense(4096, activation='relu'), name = "fc1", input = "flat1")
    model1.nodes['fc1'].set_weights(model.layers[32].get_weights())
    model1.add_node(Dropout(0.5), name = "drop1", input = "fc1")
    model1.nodes['drop1'].set_weights(model.layers[33].get_weights())
    model1.add_node(Dense(4096, activation='relu'), name = "fc2", input = "drop1")
    model1.nodes['fc2'].set_weights(model.layers[34].get_weights())
    model1.add_node(Dropout(0.5), name = "drop2", input = "fc2")
    model1.nodes['drop2'].set_weights(model.layers[35].get_weights())
    model1.add_node(Dense(1000, activation='softmax'), name = "fc3", input = "drop2")
    model1.nodes['fc3'].set_weights(model.layers[36].get_weights())
    model1.add_output(name = "out", input = "fc3")
    # print model1.nodes['fc3'].get_weights()
    return model1   

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('dog.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model1 = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model1.compile(optimizer=sgd, loss={"out" : 'categorical_crossentropy'})
    out = model1.predict({ 'input':im })
    # print model1.nodes['fc3'].get_output()
    # print get_output
    nd = np.array(out)
    # result = np.argmax(nd['out'])
    print np.argmax(out['out'])