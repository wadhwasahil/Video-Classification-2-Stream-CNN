import sys,os
import pickle

train_data={}

with open('../dataset/trainVidID.txt') as f1,open('../dataset/trainLabel.txt') as f2:
	for x,y in zip(f1,f2):
		x=x[:-1]
		y=y[:-2]
		temp = map(int,y.split(' '))
		train_data[x]=temp


with open('../dataset/train_data.pickle','wb') as f3:
	pickle.dump(train_data,f3)

print train_data