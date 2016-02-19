import cv2
import numpy as np

def getOpticalFlow(X_data_storage,Y_data_storage,filename,label):
	cap = cv2.VideoCapture(filename)
	ret, frame1 = cap.read()
	frame1 = cv2.resize(frame1, (224,224))
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	# hsv = np.zeros_like(frame1)
	# hsv[...,1] = 255

	fx = []
	fy = []
	count=0
	firstTime=1

	while(1):
		ret, frame2 = cap.read()

		if frame2==None:
		  break
		
		frame2 = cv2.resize(frame2, (224,224))
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

		flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

		horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
		vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
		horz = horz.astype('uint8')
		vert = vert.astype('uint8')

		fx.append(horz)
		fy.append(vert)
		count+=1
		if count == 10:
			flowX = np.dstack((fx[0],fx[1],fx[2],fx[3],fx[4],fx[5],fx[6],fx[7],fx[8],fx[9]))
			flowY = np.dstack((fy[0],fy[1],fy[2],fy[3],fy[4],fy[5],fy[6],fy[7],fy[8],fy[9]))
			inp = np.dstack((flowX,flowY))
			inp = np.expand_dims(inp, axis=0)
			if not firstTime:
				inputVec = np.concatenate((inputVec,inp))
			else:
				inputVec = inp
				firstTime = 0

			count = 0
			fx = []
			fy = []


		# print flow[:,:,0].shape
		# print '\n\n'
		# print flow[:,:,1].shape

		# # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		# # hsv[...,0] = ang*180/np.pi/2
		# # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		# # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
		
		print count

		# cv2.imshow('frame2',bgr)
		# k = cv2.waitKey(5) & 0xff
		# if k == 27:
		#     break
		prvs = next

	cap.release()
	cv2.destroyAllWindows()
	inputVec=np.rollaxis(inputVec,3,1)

	labels=np.tile(label,(inputVec.shape[0],1))
	labels = labels.astype('uint8')

	print inputVec.shape
	print labels.shape

	X_data_storage.append(inputVec)
	Y_data_storage.append(labels)


# f='../videos/train/-1oBvJX-sqo'
# l=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# getOpticalFlow(f,l)