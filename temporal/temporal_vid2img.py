import numpy as np
import optical_flow_prep as ofp
import sys,os
from datetime import datetime
import pickle


def writeOF():
	root = "../videos"
	path = os.path.join(root, "train")
	startTime = datetime.now()
	w=224
	h=224
	c=0
	data={}

	for path, subdirs, files in os.walk(root):
		for filename in files:
			c+=1
			count=ofp.writeOpticalFlow(path,filename,w,h,c)
			data[filename]=count

	root = "../videos"
	path = os.path.join(root, "test")
	w=224
	h=224

	for path, subdirs, files in os.walk(root):
		for filename in files:
			c+=1
			count=ofp.writeOpticalFlow(path,filename,w,h,c)
			data[filename]=count

	with open('../dataset/frame_count.pickle','wb') as f:
		pickle.dump(data,f)
	print 'Time taken for '+str(c)+'testing videos: '+str(datetime.now() - startTime)


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
				data[filename.split('_')[1].split('.')[0]+'_'+str(i)]=index+1
	with open('../dataset/temporal_train_data.pickle','wb') as f2:
		pickle.dump(data,f2)



if __name__ == "__main__":
	writeOF()
	data_prep()