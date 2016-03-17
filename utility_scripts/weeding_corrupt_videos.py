import numpy as np
import cv2
import sys,os

count=0
root = "../videos"
path = os.path.join(root, "train")

for path, subdirs, files in os.walk(root):
	for name in files:
		try:
			cap = cv2.VideoCapture('../videos/train/'+name)
		except:
			print name
			os.remove(path+'/'+name)
			count+=1

print count

count=0
root = "../videos"
path = os.path.join(root, "test")

for path, subdirs, files in os.walk(root):
	for name in files:
		try:
			cap = cv2.VideoCapture('../videos/test/'+name)
		except:
			print name
			os.remove(path+'/'+name)
			count+=1

print count