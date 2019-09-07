# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# Borrowed from davidsandberg's facenet project: https://github.com/davidsandberg/facenet
# From this directory:
#   facenet/src/align
#
# Just keep the MTCNN related stuff and removed other codes
# python package required:
#     tensorflow, opencv,numpy


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import cv2



def face_detection(img, pnet, rnet, onet):
    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    bounding_boxes, points = detect_face.detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor)

    # for b in bounding_boxes:
    #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
    #     print(b)

    temp_area = 0
    b = None
    for i, temp in enumerate(bounding_boxes):
    
        width = int(temp[3])-int(temp[1])
        longth = int(temp[2])-int(temp[0])
        area = abs(width* longth)
    
        if area > temp_area:
            temp_area = area
            b = temp
    
    box = b
    output_img = img[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])]
    cv2.imwrite('out_filename' + '_' + str(i) + '.' + 'jpg', output_img)
   
    p = points.T[0]
    landmark = []
    for i in range(5):
        landmark.append([p[i], p[i+5]])
    landmark = np.array(landmark)
    offset = np.array([int(b[0]), int(b[1])])
    size = np.array([b[2] - b[0], b[3] - b[1]])

    cv2.imwrite('output_filename.jpg',output_img)
    return output_img, landmark, offset, size, box

def face_alignment(img, landmark, offset, size):
    img_x, img_y, chanel = img.shape
    if size is not None:
        face_x , face_y = size
        dst_pts = np.array([[80.0/350.0*face_x,130.0/350.0*face_y],[250.0/350.0*face_x,130.0/350.0*face_y],[165.0/350.0*face_x,200.0/350.0*face_y],[90.0/350.0*face_x,265.0/350.0*face_y],[230.0/350.0*face_x,265.0/350.0*face_y]])
        dst_pts += offset
    else:
        dst_pts = np.array([[80.0,130.0],[250.0,130.0],[165.0,200.0],[90,265.0],[240.0,265.0]])
        landmark -=offset


    H, mask = cv2.findHomography(landmark, dst_pts, cv2.RANSAC, 5)
    
    if size is not None:
        imgReg = cv2.warpPerspective(img, H, (img_y, img_x))
    else:
        imgReg = cv2.warpPerspective(img, H, (350, 350))

    return imgReg 

def total_algnment(img, pnet, rnet, onet,fname_):
    imgReg_, landmark_, offset_, size_, box_ = face_detection(img, pnet, rnet, onet)
    img_return = face_alignment(img, landmark_, offset_, size_)
    
    out_image = img_return[int(box_[1]) : int(box_[3]), int(box_[0]) : int(box_[2])]
    # w,h,out_chanel = np.shape(out_image)
    # print(out_chanel,fname_)
    out_image = cv2.resize(out_image,(299,299))
    cv2.imwrite(fname_, out_image)

if __name__ == '__main__':
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    train_cats_dirs = {'Happy','./Disgust/','./Fear/','./Sad/', 'Neutral','./Surprise/','./Angry/'}
    # train_cats_dirs = {'Angry'}
    for train_cats_dir in train_cats_dirs:
        print("Start " + train_cats_dir)
        fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
        for fname_ in fnames:
            img = cv2.imread(fname_)
            total_algnment(img, pnet, rnet, onet, fname_)

    
    
