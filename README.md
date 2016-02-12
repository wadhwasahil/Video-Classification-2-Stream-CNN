# Video Classification using Two Stream CNNs

We use a spatial and a temporal stream with VGG-16 and CNN-M respectively for modeling video information.
LSTMs are stacked on top of the CNNs for modeling long term dependencies between video frames.
For more information, see these papers:

[Two-Stream Convolutional Networks for Action Recognition in Videos](http://arxiv.org/pdf/1406.2199v2.pdf)
[Fusing Multi-Stream Deep Networks for Video Classification](http://arxiv.org/pdf/1509.06086v2.pdf)
[Modeling Spatial-Temporal Clues in a Hybrid Deep Learning Framework for Video Classification](http://arxiv.org/pdf/1504.01561v1.pdf)
[Towards Good Practices for Very Deep Two-Stream ConvNets](http://arxiv.org/pdf/1507.02159v1.pdf)