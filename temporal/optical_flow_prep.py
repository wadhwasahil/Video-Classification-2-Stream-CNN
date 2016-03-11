import cv2
import numpy as np

def stackOpticalFlow(filename,w,h,label):
	pass
	# firstTime=1
	# fx = []
	# fy = []
	# fx.append(horz)
	# fy.append(vert)
	# count+=1
	# if count == 10:
	# 	flowX = np.dstack((fx[0],fx[1],fx[2],fx[3],fx[4],fx[5],fx[6],fx[7],fx[8],fx[9]))
	# 	flowY = np.dstack((fy[0],fy[1],fy[2],fy[3],fy[4],fy[5],fy[6],fy[7],fy[8],fy[9]))
	# 	inp = np.dstack((flowX,flowY))
	# 	inp = np.expand_dims(inp, axis=0)
	# 	if not firstTime:
	# 		inputVec = np.concatenate((inputVec,inp))
	# 	else:
	# 		inputVec = inp
	# 		firstTime = 0

	# 	count = 0
	# 	fx = []
	# 	fy = []

	# inputVec=np.rollaxis(inputVec,3,1)
	# labels=np.tile(label,(inputVec.shape[0],1))
	# labels = labels.astype('uint8')

def writeOpticalFlow(path,filename,w,h,c):
	cap = cv2.VideoCapture(path+'/'+filename)
	ret, frame1 = cap.read()
	frame1 = cv2.resize(frame1, (w,h))
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

	count=0

	while(1):
		ret, frame2 = cap.read()

		if frame2==None:
			break

		count+=1
		print str(c)+'-'+str(count)

		frame2 = cv2.resize(frame2, (w,h))
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
		vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
		horz = horz.astype('uint8')
		vert = vert.astype('uint8')

		cv2.imwrite('./optical_flow_images/h'+str(count)+'_'+filename+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
		cv2.imwrite('./optical_flow_images/v'+str(count)+'_'+filename+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
		
		prvs = next

	cap.release()
	cv2.destroyAllWindows()
	return count