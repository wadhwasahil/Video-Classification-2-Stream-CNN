# This program reads the weights from a hd5 file

import h5py
import numpy as np
import theano
from ast import literal_eval
f = h5py.File('vgg16_weights.h5', 'r')
a  = []
cnt  = 0
# for k in range(f.attrs['nb_layers']):
#         g = f['layer_{}'.format(k)]
#         weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#         cnt = cnt + 1
data = f['/']['layer_1']['param_0']
print np.array(data)
