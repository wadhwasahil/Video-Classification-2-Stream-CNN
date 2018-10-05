# Video Classification using Two Stream CNNs

We use a spatial and a temporal stream with VGG-16 and CNN-M respectively for modeling video information.
LSTMs are stacked on top of the CNNs for modeling long term dependencies between video frames.
For more information, see these papers:

[Two-Stream Convolutional Networks for Action Recognition in Videos](http://arxiv.org/pdf/1406.2199v2.pdf)

[Fusing Multi-Stream Deep Networks for Video Classification](http://arxiv.org/pdf/1509.06086v2.pdf)

[Modeling Spatial-Temporal Clues in a Hybrid Deep Learning Framework for Video Classification](http://arxiv.org/pdf/1504.01561v1.pdf)

[Towards Good Practices for Very Deep Two-Stream ConvNets](http://arxiv.org/pdf/1507.02159v1.pdf)

***

Here are the steps to run the project on CCV dataset:

## Creating a virtual environment

First create a directory named env and then run the following inside the directory. This will create a virtual environment. Assuming we create a requirements.txt file to help install modules that are needed in the project.
`$ mkdir env`
`$ cd env`
`$ virtualenv venv-video-classification `
`$ source  env-video-classification\bin\activate`
`$ cd ..`
`$ pip install requirements.txt` 


## Setting up the DataSet
1. Get the YouTube data, remove broken videos and negative instances and finally create a pickle file of the dataset by running scripts from the utility_scripts folder


2. Temporal Stream (in the temporal folder):
  1. Run temporal_vid2img to create optical flow frames and the related files
  2. Run temporal_stream_cnn to start with the temporal stream training


3. Spatial Stream (in the spatial folder):
  1. Run the spatial_vid2img to create static frames and related files
  2. Download the vgg16_weights.h5 file from [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) and put it in the spatial folder

  3. Run spatial_stream_cnn to start with the spatial stream training

4. Temporal Stream LSTM: 
Will soon update the code

5. Spatial Stream LSTM: 
Will soon update the code

***
