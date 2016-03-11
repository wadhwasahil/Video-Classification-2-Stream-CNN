import numpy as np
import sys,os
import pickle

def data_prep():
	with open('../dataset/frame_count.pickle','rb') as f:
		frame_count=pickle.load(f)
	root = './optical_flow_images'
	path = os.path.join(root, '')
	data={}

	for path, subdirs, files in os.walk(root):
		for filename in files:
			fc=frame_count[filename.split('_')[1].split('.')[0]]
			for i,j in enumerate(train_data[filename.split('_')[1].split('.')[0]]):
				if j:
					index=i
					break
			for i in range(1,(fc/10)+1):
				data[filename+'_'+str(i)]=index+1
	with open('../dataset/temporal_train_data.pickle','wb') as f2:
		pickle.dump(data,f2)
			

def vectorize():
	pass

if __name__ == "__main__":
	data_prep()
