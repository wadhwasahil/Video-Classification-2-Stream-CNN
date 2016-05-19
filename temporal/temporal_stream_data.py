import numpy as np
import sys,os
import pickle
import optical_flow_prep as ofp
import gc


def stackOF(chunk,img_rows,img_cols):
	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	X_train,Y_train=ofp.stackOpticalFlow(chunk,temporal_train_data,img_rows,img_cols)
	gc.collect()
	return (X_train,Y_train)
