#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  cv2
import numpy as np
from keras.models import load_model
from keras.applications import VGG16, InceptionResNetV2

import sys
import argparse
import tensorflow as tf
import numpy as np
import detect_face

# In[2]:


EMOTION_DICT = {1:"ANGRY", 2:"DISGUST", 3:"FEAR", 4:"HAPPY", 5:"NEUTRAL", 6:"SAD", 7:"SURPRISE"}
model_VGG = InceptionResNetV2(weights='imagenet', include_top=False)
model_top = load_model("../Data/Model_Save/model.h5")

def face_detection(img, pnet, rnet, onet):
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    bounding_boxes, points = detect_face.detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor)

    # for b in bounding_boxes:
    #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
    #     print(b)

    for i, b in enumerate(bounding_boxes):
        box = b
        output_img = img[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])]
        cv2.imwrite('out_filename' + '_' + str(i) + '.' + 'jpg', output_img)
   
    if points == []:
        return ()
    return box


# In[3]:

def face_det_crop_resize(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]
        cv2.imwrite(path, cv2.resize(face_clip, (299, 299)))


from facedetect_mtcnn import total_algnment
sess = tf.Session()
pnet, rnet, onet = detect_face.create_mtcnn(sess, None) 

def return_prediction(path):
    #converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, gray)
    
    #detect face in image, crop it then resize it then save it
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    # face_det_crop_resize(path)

    total_algnment(img,pnet,rnet,onet, path)
    #read the processed image then make prediction and display the result
    read_image = cv2.imread(path)
    if read_image.shape != (299,299,3):
        return "NO FACE"
    read_image = read_image.reshape(1, read_image.shape[0], read_image.shape[1], read_image.shape[2])
    read_image_final = read_image/255.0
    VGG_Pred = model_VGG.predict(read_image_final)
    VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1]*VGG_Pred.shape[2]*VGG_Pred.shape[3])
    top_pred = model_top.predict(VGG_Pred)
    emotion_label = top_pred[0].argmax() + 1
    return EMOTION_DICT[emotion_label]


# In[4]:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

def rerun(text, cap):
    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Last Emotion was "+str(text), (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(img, "Press SPACE: FOR EMOTION", (5,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
        cv2.putText(img, "Hold Q: To Quit", (460,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # faces = [face_detection(img, pnet, rnet, onet)]
        # print(faces)
        # for x1,x2,y1,y2,precision in faces:
        #     cv2.rectangle(img, (int(x1),int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test.jpg", img)
            text = return_prediction("test.jpg")
            first_run(text, cap)
            break
            
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# In[5]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture(0)
prototxt_path = os.path.join('./models/deploy.prototxt')
caffemodel_path = os.path.join('./models/weights.caffemodel')

    # Read the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
def first_run(text, cap):
    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Last Emotion was "+str(text), (95,30), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(img, "Press SPACE: FOR EMOTION", (5,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
        cv2.putText(img, "Hold Q: To Quit", (460,470), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # for x,y,w,h in faces:
        #     cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == ord(' '):
            cv2.imwrite("test.jpg", img)
            text = return_prediction("test.jpg")
            rerun(text, cap)
            break
            
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# In[6]:


first_run("None", cap)


# In[ ]:




