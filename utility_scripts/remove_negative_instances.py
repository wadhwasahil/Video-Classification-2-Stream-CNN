import numpy as np
import pickle
import sys,os

root = "../videos"
path = os.path.join(root, "train")
deleteList=[]
count=0

with open('../dataset/train_data.pickle','rb') as f3:
	train_data=pickle.load(f3)

for vid in train_data.keys():
	if all(v == 0 for v in train_data[vid]):
		deleteList.append(vid)

print len(deleteList)

for i in deleteList:
	if os.path.isfile(path+'/'+i):
		os.remove(path+'/'+i)
		count+=1

print count

root = "../videos"
path = os.path.join(root, "test")
deleteList=[]
count=0

with open('../dataset/test_data.pickle','rb') as f3:
	test_data=pickle.load(f3)

for vid in test_data.keys():
	if all(v == 0 for v in test_data[vid]):
		deleteList.append(vid)

print len(deleteList)

for i in deleteList:
	if os.path.isfile(path+'/'+i):
		os.remove(path+'/'+i)
		count+=1

print count