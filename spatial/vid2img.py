import numpy as np
import cv2
import sys
import os
import pickle

root = './Videos'
path = os.path.join(root, "")

with open('../dataset/train_data.pickle', 'rb') as f:
    var1 = pickle.load(f)
<<<<<<< HEAD
ptr = dict()    
=======
ptr = dict()
>>>>>>> e762869718b115d5b8ae33c0db466c3724152eec
for path, subdirs, files in os.walk(root):
    for filename in files:
        full_path = path + '/' + filename
        cap = cv2.VideoCapture(full_path)
        cnt = 1
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame == None:
                break
            # cv2.imshow('frame', frame)

            # Image Path
            fpath = 'Data_Images/'
            vid_name = filename.split('.')[0]
            img_path = fpath + vid_name + '_{}.jpg'.format(cnt)
            
            img_name = vid_name + '_{}'.format(cnt)
            cv2.imwrite(img_path, frame)
            cnt = cnt + 1
            ptr[img_name.split('.')[0]] = var1[vid_name].index(1) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

with open('../dataset/train_spatial_data.pickle', 'wb') as g:
    pickle.dump(ptr, g)
