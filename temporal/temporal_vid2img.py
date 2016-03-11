import numpy as np
import optical_flow_prep as ofp
import sys,os
from datetime import datetime


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
	with open('../dataset/frame_count.pickle','wb') as f:
		pickle.dump(data,f)
	print 'Time taken for '+str(c)+'videos: '+str(datetime.now() - startTime)

writeOF()