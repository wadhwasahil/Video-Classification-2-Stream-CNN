import numpy as np
import optical_flow_prep as ofp
import sys,os
import tables
import pickle
from datetime import datetime

startTime = datetime.now()


root = "../videos"
path = os.path.join(root, "train")
count=0

with open('../dataset/train_data.pickle','rb') as f3:
	train_data=pickle.load(f3)


hdf5_path = "../dataset/training_data_final.hdf5"
hdf5_file = tables.openFile(hdf5_path, mode='w')
atom = tables.UInt8Atom()
X_data_storage = hdf5_file.createEArray(hdf5_file.root, 'X_train',atom,shape=(0,20,224,224),expectedrows=200)
Y_data_storage = hdf5_file.createEArray(hdf5_file.root, 'Y_train',atom,shape=(0,20),expectedrows=200)

for path, subdirs, files in os.walk(root):
	for filename in files:
		count+=1
		if count>3:
			print datetime.now() - startTime
			sys.exit()
		label=train_data[filename]
		ofp.getOpticalFlow(X_data_storage,Y_data_storage,path+'/'+filename,label)
print datetime.now() - startTime
hdf5_file.close()