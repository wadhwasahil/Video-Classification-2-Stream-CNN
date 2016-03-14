import numpy as np
import sys,os
import pickle
import optical_flow_prep as ofp

def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def stackOF():
	chunk_size=5000

	with open('../dataset/temporal_train_data.pickle','rb') as f1:
		temporal_train_data=pickle.load(f1)

	chunk=chunks(temporal_train_data.keys(),chunk_size)
	for blocks in chunk:
		X_train,Y_train=ofp.stackOpticalFlow(blocks,temporal_train_data)
		yield (X_train,Y_train)
