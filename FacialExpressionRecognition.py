#!/usr/bin/env python
# coding: utf-8

# # Real Time Facial Expression Recognition

# ## Description
# Computer animated agents and robots bring new dimension in human computer interaction which makes it vital as how computers can affect our social life in day-to-day activities. Face to face communication is a real-time process operating at a a time scale in the order of milliseconds. The level of uncertainty at this time scale is considerable, making it necessary for humans and machines to rely on sensory rich perceptual primitives rather than slow symbolic inference processes.<br><br>
# In this project we are presenting the real time facial expression recognition of seven most basic human expressions: ANGER, DISGUST, FEAR, HAPPY, NEUTRAL SAD, SURPRISE.<br><br>
# This model can be used for prediction of expressions of both still images and real time video. However, in both the cases we have to provide image to the model. In case of real time video the image should be taken at any point in time and feed it to the model for prediction of expression. The system automatically detects face using HAAR cascade then its crops it and resize the image to a specific size and give it to the model for prediction. The model will generate seven probability values corresponding to seven expressions. The highest probability value to the corresponding expression will be the predicted expression for that image.<br><br>

# ## Business Problem
# However, our goal here is to predict the human expressions, but we have trained our model on both human and animated images. Since, we had only approx 1500 human images which are very less to make a good model, so we took approximately 9000 animated images and leverage those animated images for training the model and ultimately do the prediction of expressions on human images.<br><br> 
# For better prediction we have decided to keep the size of each image <b>350$*$350</b>.<br><br>
# <b>For any image our goal is to predict the expression of the face in that image out of seven basic human expression</b>

# ## Problem Statement
# <br>
# <B>CLASSIFY THE EXPRESSION OF FACE IN IMAGE OUT OF SEVEN BASIC HUMAN EXPRESSION</B>

# ## Performance Metric
# This is a multi-class classification problem with 7 different classes, so we have considered three performance metrics:<br>
# 1. Multi-Class Log-loss
# 2. Accuracy
# 3. Confusion Metric

# ## Source Data
# We have downloaded data from 4 different sources.<br>
# 1. Human Images Source-1: http://www.consortium.ri.cmu.edu/ckagree/
# 2. Human Images Source-2: http://app.visgraf.impa.br/database/faces/
# 3. Human Images Source-3: http://www.kasrl.org/jaffe.html
# 4. Animated Images Source: https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html

# ## Real-World Business Objective & Constraints
# 1. Low-latency is required.
# 2. Interpretability is important for still images but not in real time. For still images, probability of predicted expressions can be given.
# 3. Errors are not costly.

# ## Y- Encoded Labels
# __Angry--1__<br>
# __Disgust --2__<br>
# __Fear--3__<br>
# __Happy--4__<br>
# __Neutral--5__<br>
# __Sad--6__<br>
# __Surprise--7__

# ## Mapping real-world to ML Problem

# In[1]:


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.applications import VGG16, InceptionResNetV2
from sklearn.metrics import accuracy_score, confusion_matrix


# ## 1. Reading the Data of Human Images

# ### Angry

# In[40]:


human_angry_test = glob.glob("../Data/Human/test/Angry/*")
human_angry_train = glob.glob("../Data/Human/train/Angry/*")
human_angry_cv = glob.glob("../Data/Human/cv/Angry/*")
# human_angry.remove('../Data/Human/Angry\\Thumbs.db')
print("Number of images in Angry emotion = "+str(len(human_angry_test)+len(human_angry_cv)+len(human_angry_train)))


# In[84]:


human_angry_test_folderName = [str(i.split("\\")[0])+"/" for i in human_angry_test]
human_angry_test_imageName = [str(i.split("\\")[1]) for i in human_angry_test]
human_angry_test_emotion = [["Angry"]*len(human_angry_test)][0]
human_angry_test_label = [1]*len(human_angry_test)

human_angry_train_folderName = [str(i.split("\\")[0])+"/" for i in human_angry_train]
human_angry_train_imageName = [str(i.split("\\")[1]) for i in human_angry_train]
human_angry_train_emotion = [["Angry"]*len(human_angry_train)][0]
human_angry_train_label = [1]*len(human_angry_train)

human_angry_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_angry_cv]
human_angry_cv_imageName = [str(i.split("\\")[1]) for i in human_angry_cv]
human_angry_cv_emotion = [["Angry"]*len(human_angry_cv)][0]
human_angry_cv_label = [1]*len(human_angry_cv)

# In[86]:


df_angry_test = pd.DataFrame()
df_angry_test["folderName"] = human_angry_test_folderName
df_angry_test["imageName"] = human_angry_test_imageName
df_angry_test["Emotion"] = human_angry_test_emotion
df_angry_test["Labels"] = human_angry_test_label
df_angry_test.head()

df_angry_train = pd.DataFrame()
df_angry_train["folderName"] = human_angry_train_folderName
df_angry_train["imageName"] = human_angry_train_imageName
df_angry_train["Emotion"] = human_angry_train_emotion
df_angry_train["Labels"] = human_angry_train_label
df_angry_train.head()

df_angry_cv = pd.DataFrame()
df_angry_cv["folderName"] = human_angry_cv_folderName
df_angry_cv["imageName"] = human_angry_cv_imageName
df_angry_cv["Emotion"] = human_angry_cv_emotion
df_angry_cv["Labels"] = human_angry_cv_label
df_angry_cv.head()

# ### Disgust

# In[87]:


human_disgust_test = glob.glob("../Data/Human/test/Disgust/*")
human_disgust_train = glob.glob("../Data/Human/train/Disgust/*")
human_disgust_cv = glob.glob("../Data/Human/cv/Disgust/*")
# human_disgust.remove('../Data/Human/disgust\\Thumbs.db')
print("Number of images in disgust emotion = "+str(len(human_disgust_test)+len(human_disgust_cv)+len(human_disgust_train)))


# In[84]:


human_disgust_test_folderName = [str(i.split("\\")[0])+"/" for i in human_disgust_test]
human_disgust_test_imageName = [str(i.split("\\")[1]) for i in human_disgust_test]
human_disgust_test_emotion = [["Disgust"]*len(human_disgust_test)][0]
human_disgust_test_label = [1]*len(human_disgust_test)

human_disgust_train_folderName = [str(i.split("\\")[0])+"/" for i in human_disgust_train]
human_disgust_train_imageName = [str(i.split("\\")[1]) for i in human_disgust_train]
human_disgust_train_emotion = [["Disgust"]*len(human_disgust_train)][0]
human_disgust_train_label = [1]*len(human_disgust_train)

human_disgust_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_disgust_cv]
human_disgust_cv_imageName = [str(i.split("\\")[1]) for i in human_disgust_cv]
human_disgust_cv_emotion = [["Disgust"]*len(human_disgust_cv)][0]
human_disgust_cv_label = [1]*len(human_disgust_cv)

# In[86]:


df_disgust_test = pd.DataFrame()
df_disgust_test["folderName"] = human_disgust_test_folderName
df_disgust_test["imageName"] = human_disgust_test_imageName
df_disgust_test["Emotion"] = human_disgust_test_emotion
df_disgust_test["Labels"] = human_disgust_test_label
df_disgust_test.head()

df_disgust_train = pd.DataFrame()
df_disgust_train["folderName"] = human_disgust_train_folderName
df_disgust_train["imageName"] = human_disgust_train_imageName
df_disgust_train["Emotion"] = human_disgust_train_emotion
df_disgust_train["Labels"] = human_disgust_train_label
df_disgust_train.head()

df_disgust_cv = pd.DataFrame()
df_disgust_cv["folderName"] = human_disgust_cv_folderName
df_disgust_cv["imageName"] = human_disgust_cv_imageName
df_disgust_cv["Emotion"] = human_disgust_cv_emotion
df_disgust_cv["Labels"] = human_disgust_cv_label
df_disgust_cv.head()


# ### Fear

# In[117]:


human_fear_test = glob.glob("../Data/Human/test/Fear/*")
human_fear_train = glob.glob("../Data/Human/train/Fear/*")
human_fear_cv = glob.glob("../Data/Human/cv/Fear/*")
# human_fear.remove('../Data/Human/fear\\Thumbs.db')
print("Number of images in fear emotion = "+str(len(human_fear_test)+len(human_fear_cv)+len(human_fear_train)))


# In[84]:


human_fear_test_folderName = [str(i.split("\\")[0])+"/" for i in human_fear_test]
human_fear_test_imageName = [str(i.split("\\")[1]) for i in human_fear_test]
human_fear_test_emotion = [["Fear"]*len(human_fear_test)][0]
human_fear_test_label = [1]*len(human_fear_test)

human_fear_train_folderName = [str(i.split("\\")[0])+"/" for i in human_fear_train]
human_fear_train_imageName = [str(i.split("\\")[1]) for i in human_fear_train]
human_fear_train_emotion = [["Fear"]*len(human_fear_train)][0]
human_fear_train_label = [1]*len(human_fear_train)

human_fear_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_fear_cv]
human_fear_cv_imageName = [str(i.split("\\")[1]) for i in human_fear_cv]
human_fear_cv_emotion = [["Fea"]*len(human_fear_cv)][0]
human_fear_cv_label = [1]*len(human_fear_cv)

# In[86]:


df_fear_test = pd.DataFrame()
df_fear_test["folderName"] = human_fear_test_folderName
df_fear_test["imageName"] = human_fear_test_imageName
df_fear_test["Emotion"] = human_fear_test_emotion
df_fear_test["Labels"] = human_fear_test_label
df_fear_test.head()

df_fear_train = pd.DataFrame()
df_fear_train["folderName"] = human_fear_train_folderName
df_fear_train["imageName"] = human_fear_train_imageName
df_fear_train["Emotion"] = human_fear_train_emotion
df_fear_train["Labels"] = human_fear_train_label
df_fear_train.head()

df_fear_cv = pd.DataFrame()
df_fear_cv["folderName"] = human_fear_cv_folderName
df_fear_cv["imageName"] = human_fear_cv_imageName
df_fear_cv["Emotion"] = human_fear_cv_emotion
df_fear_cv["Labels"] = human_fear_cv_label
df_fear_cv.head()

# ### Happy

# In[149]:


human_happy_test = glob.glob("../Data/Human/test/Happy/*")
human_happy_train = glob.glob("../Data/Human/train/Happy/*")
human_happy_cv = glob.glob("../Data/Human/cv/Happy/*")
# human_happy.remove('../Data/Human/happy\\Thumbs.db')
print("Number of images in happy emotion = "+str(len(human_happy_test)+len(human_happy_cv)+len(human_happy_train)))


# In[84]:


human_happy_test_folderName = [str(i.split("\\")[0])+"/" for i in human_happy_test]
human_happy_test_imageName = [str(i.split("\\")[1]) for i in human_happy_test]
human_happy_test_emotion = [["Happy"]*len(human_happy_test)][0]
human_happy_test_label = [1]*len(human_happy_test)

human_happy_train_folderName = [str(i.split("\\")[0])+"/" for i in human_happy_train]
human_happy_train_imageName = [str(i.split("\\")[1]) for i in human_happy_train]
human_happy_train_emotion = [["Happy"]*len(human_happy_train)][0]
human_happy_train_label = [1]*len(human_happy_train)

human_happy_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_happy_cv]
human_happy_cv_imageName = [str(i.split("\\")[1]) for i in human_happy_cv]
human_happy_cv_emotion = [["Happy"]*len(human_happy_cv)][0]
human_happy_cv_label = [1]*len(human_happy_cv)

# In[86]:


df_happy_test = pd.DataFrame()
df_happy_test["folderName"] = human_happy_test_folderName
df_happy_test["imageName"] = human_happy_test_imageName
df_happy_test["Emotion"] = human_happy_test_emotion
df_happy_test["Labels"] = human_happy_test_label
df_happy_test.head()

df_happy_train = pd.DataFrame()
df_happy_train["folderName"] = human_happy_train_folderName
df_happy_train["imageName"] = human_happy_train_imageName
df_happy_train["Emotion"] = human_happy_train_emotion
df_happy_train["Labels"] = human_happy_train_label
df_happy_train.head()

df_happy_cv = pd.DataFrame()
df_happy_cv["folderName"] = human_happy_cv_folderName
df_happy_cv["imageName"] = human_happy_cv_imageName
df_happy_cv["Emotion"] = human_happy_cv_emotion
df_happy_cv["Labels"] = human_happy_cv_label
df_happy_cv.head()


# ### Neutral

# In[154]:


human_neutral_test = glob.glob("../Data/Human/test/Neutral/*")
human_neutral_train = glob.glob("../Data/Human/train/Neutral/*")
human_neutral_cv = glob.glob("../Data/Human/cv/Neutral/*")
# human_neutral.remove('../Data/Human/neutral\\Thumbs.db')
print("Number of images in neutral emotion = "+str(len(human_neutral_test)+len(human_neutral_cv)+len(human_neutral_train)))


# In[84]:


human_neutral_test_folderName = [str(i.split("\\")[0])+"/" for i in human_neutral_test]
human_neutral_test_imageName = [str(i.split("\\")[1]) for i in human_neutral_test]
human_neutral_test_emotion = [["Neutral"]*len(human_neutral_test)][0]
human_neutral_test_label = [1]*len(human_neutral_test)

human_neutral_train_folderName = [str(i.split("\\")[0])+"/" for i in human_neutral_train]
human_neutral_train_imageName = [str(i.split("\\")[1]) for i in human_neutral_train]
human_neutral_train_emotion = [["Neutral"]*len(human_neutral_train)][0]
human_neutral_train_label = [1]*len(human_neutral_train)

human_neutral_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_neutral_cv]
human_neutral_cv_imageName = [str(i.split("\\")[1]) for i in human_neutral_cv]
human_neutral_cv_emotion = [["Neutral"]*len(human_neutral_cv)][0]
human_neutral_cv_label = [1]*len(human_neutral_cv)

# In[86]:


df_neutral_test = pd.DataFrame()
df_neutral_test["folderName"] = human_neutral_test_folderName
df_neutral_test["imageName"] = human_neutral_test_imageName
df_neutral_test["Emotion"] = human_neutral_test_emotion
df_neutral_test["Labels"] = human_neutral_test_label
df_neutral_test.head()

df_neutral_train = pd.DataFrame()
df_neutral_train["folderName"] = human_neutral_train_folderName
df_neutral_train["imageName"] = human_neutral_train_imageName
df_neutral_train["Emotion"] = human_neutral_train_emotion
df_neutral_train["Labels"] = human_neutral_train_label
df_neutral_train.head()

df_neutral_cv = pd.DataFrame()
df_neutral_cv["folderName"] = human_neutral_cv_folderName
df_neutral_cv["imageName"] = human_neutral_cv_imageName
df_neutral_cv["Emotion"] = human_neutral_cv_emotion
df_neutral_cv["Labels"] = human_neutral_cv_label
df_neutral_cv.head()


# ### Sad

# In[181]:


human_sad_test = glob.glob("../Data/Human/test/Sad/*")
human_sad_train = glob.glob("../Data/Human/train/Sad/*")
human_sad_cv = glob.glob("../Data/Human/cv/Sad/*")
# human_sad.remove('../Data/Human/sad\\Thumbs.db')
print("Number of images in sad emotion = "+str(len(human_sad_test)+len(human_sad_cv)+len(human_sad_train)))


# In[84]:


human_sad_test_folderName = [str(i.split("\\")[0])+"/" for i in human_sad_test]
human_sad_test_imageName = [str(i.split("\\")[1]) for i in human_sad_test]
human_sad_test_emotion = [["Sad"]*len(human_sad_test)][0]
human_sad_test_label = [1]*len(human_sad_test)

human_sad_train_folderName = [str(i.split("\\")[0])+"/" for i in human_sad_train]
human_sad_train_imageName = [str(i.split("\\")[1]) for i in human_sad_train]
human_sad_train_emotion = [["Sad"]*len(human_sad_train)][0]
human_sad_train_label = [1]*len(human_sad_train)

human_sad_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_sad_cv]
human_sad_cv_imageName = [str(i.split("\\")[1]) for i in human_sad_cv]
human_sad_cv_emotion = [["Sad"]*len(human_sad_cv)][0]
human_sad_cv_label = [1]*len(human_sad_cv)

# In[86]:


df_sad_test = pd.DataFrame()
df_sad_test["folderName"] = human_sad_test_folderName
df_sad_test["imageName"] = human_sad_test_imageName
df_sad_test["Emotion"] = human_sad_test_emotion
df_sad_test["Labels"] = human_sad_test_label
df_sad_test.head()

df_sad_train = pd.DataFrame()
df_sad_train["folderName"] = human_sad_train_folderName
df_sad_train["imageName"] = human_sad_train_imageName
df_sad_train["Emotion"] = human_sad_train_emotion
df_sad_train["Labels"] = human_sad_train_label
df_sad_train.head()

df_sad_cv = pd.DataFrame()
df_sad_cv["folderName"] = human_sad_cv_folderName
df_sad_cv["imageName"] = human_sad_cv_imageName
df_sad_cv["Emotion"] = human_sad_cv_emotion
df_sad_cv["Labels"] = human_sad_cv_label
df_sad_cv.head()


# ### Surprise

# In[231]:


human_surprise_test = glob.glob("../Data/Human/test/Surprise/*")
human_surprise_train = glob.glob("../Data/Human/train/Surprise/*")
human_surprise_cv = glob.glob("../Data/Human/cv/Surprise/*")
# human_surprise.remove('../Data/Human/surprise\\Thumbs.db')
print("Number of images in surprise emotion = "+str(len(human_surprise_test)+len(human_surprise_cv)+len(human_surprise_train)))


# In[84]:


human_surprise_test_folderName = [str(i.split("\\")[0])+"/" for i in human_surprise_test]
human_surprise_test_imageName = [str(i.split("\\")[1]) for i in human_surprise_test]
human_surprise_test_emotion = [["Surprise"]*len(human_surprise_test)][0]
human_surprise_test_label = [1]*len(human_surprise_test)

human_surprise_train_folderName = [str(i.split("\\")[0])+"/" for i in human_surprise_train]
human_surprise_train_imageName = [str(i.split("\\")[1]) for i in human_surprise_train]
human_surprise_train_emotion = [["Surprise"]*len(human_surprise_train)][0]
human_surprise_train_label = [1]*len(human_surprise_train)

human_surprise_cv_folderName = [str(i.split("\\")[0])+"/" for i in human_surprise_cv]
human_surprise_cv_imageName = [str(i.split("\\")[1]) for i in human_surprise_cv]
human_surprise_cv_emotion = [["Surprise"]*len(human_surprise_cv)][0]
human_surprise_cv_label = [1]*len(human_surprise_cv)

# In[86]:


df_surprise_test = pd.DataFrame()
df_surprise_test["folderName"] = human_surprise_test_folderName
df_surprise_test["imageName"] = human_surprise_test_imageName
df_surprise_test["Emotion"] = human_surprise_test_emotion
df_surprise_test["Labels"] = human_surprise_test_label
df_surprise_test.head()

df_surprise_train = pd.DataFrame()
df_surprise_train["folderName"] = human_surprise_train_folderName
df_surprise_train["imageName"] = human_surprise_train_imageName
df_surprise_train["Emotion"] = human_surprise_train_emotion
df_surprise_train["Labels"] = human_surprise_train_label
df_surprise_train.head()

df_surprise_cv = pd.DataFrame()
df_surprise_cv["folderName"] = human_surprise_cv_folderName
df_surprise_cv["imageName"] = human_surprise_cv_imageName
df_surprise_cv["Emotion"] = human_surprise_cv_emotion
df_surprise_cv["Labels"] = human_surprise_cv_label
df_surprise_cv.head()


# In[255]:


length_test = df_angry_test.shape[0] + df_disgust_test.shape[0] + df_fear_test.shape[0] + df_happy_test.shape[0] + df_neutral_test.shape[0] + df_sad_test.shape[0] + df_surprise_test.shape[0]
length_train = df_angry_train.shape[0] + df_disgust_train.shape[0] + df_fear_train.shape[0] + df_happy_train.shape[0] + df_neutral_train.shape[0] + df_sad_train.shape[0] + df_surprise_train.shape[0]
length_cv = df_angry_cv.shape[0] + df_disgust_cv.shape[0] + df_fear_cv.shape[0] + df_happy_cv.shape[0] + df_neutral_cv.shape[0] + df_sad_cv.shape[0] + df_surprise_cv.shape[0]
print("Total number of images in all the emotions = "+str(length_cv+length_train+length_test))


# ### Concatenating all dataframes

# In[256]:


frames_test = [df_angry_test, df_disgust_test, df_fear_test, df_happy_test, df_neutral_test, df_sad_test, df_surprise_test]
Final_human_test = pd.concat(frames_test)

frames_train = [df_angry_train, df_disgust_train, df_fear_train, df_happy_train, df_neutral_train, df_sad_train, df_surprise_train]
Final_human_train = pd.concat(frames_train)

frames_cv = [df_angry_cv, df_disgust_cv, df_fear_cv, df_happy_cv, df_neutral_cv, df_sad_cv, df_surprise_cv]
Final_human_cv = pd.concat(frames_cv)

# In[261]:


Final_human_test.reset_index(inplace = True, drop = True)
Final_human_test = Final_human_test.sample(frac = 1.0)   #shuffling the dataframe
Final_human_test.reset_index(inplace = True, drop = True)
Final_human_test.head()

Final_human_train.reset_index(inplace = True, drop = True)
Final_human_train = Final_human_train.sample(frac = 1.0)   #shuffling the dataframe
Final_human_train.reset_index(inplace = True, drop = True)
Final_human_train.head()

Final_human_cv.reset_index(inplace = True, drop = True)
Final_human_cv = Final_human_cv.sample(frac = 1.0)   #shuffling the dataframe
Final_human_cv.reset_index(inplace = True, drop = True)
Final_human_cv.head()

# ## 2. Train, CV and Test Split for Human Images

# In[307]:

'''hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
df_human_train_data, df_human_test = train_test_split(Final_human, stratify=Final_human["Labels"], test_size = 0.197860)
df_human_train, df_human_cv = train_test_split(df_human_train_data, stratify=df_human_train_data["Labels"], test_size = 0.166666)
df_human_train.shape, df_human_cv.shape, df_human_test.shape 
''' 
df_human_test = Final_human_test
df_human_train = Final_human_train
df_human_cv = Final_human_cv

# In[308]:


df_human_train.reset_index(inplace = True, drop = True)
df_human_train.to_pickle("../Data/Dataframes/Human/df_human_train.pkl")

df_human_cv.reset_index(inplace = True, drop = True)
df_human_cv.to_pickle("../Data/Dataframes/Human/df_human_cv.pkl")

df_human_test.reset_index(inplace = True, drop = True)
df_human_test.to_pickle("../Data/Dataframes/Human/df_human_test.pkl")


# In[2]:


df_human_train = pd.read_pickle("../Data/Dataframes/Human/df_human_train.pkl")
df_human_train.head()


# In[3]:


print(df_human_train.shape)


# In[4]:


df_human_cv = pd.read_pickle("../Data/Dataframes/Human/df_human_cv.pkl")
df_human_cv.head()


# In[5]:


print(df_human_cv.shape)


# In[6]:


df_human_test = pd.read_pickle("../Data/Dataframes/Human/df_human_test.pkl")
df_human_test.head()


# In[7]:


print(df_human_test.shape)


# ## 3. Analysing Data of Human Images
# ### Distribution of class labels in Train, CV and Test

# In[354]:


df_temp_train = df_human_train.sort_values(by = "Labels", inplace = False)
df_temp_cv = df_human_cv.sort_values(by = "Labels", inplace = False)
df_temp_test = df_human_test.sort_values(by = "Labels", inplace = False)

TrainData_distribution = df_human_train["Emotion"].value_counts().sort_index()
CVData_distribution = df_human_cv["Emotion"].value_counts().sort_index()
TestData_distribution = df_human_test["Emotion"].value_counts().sort_index()

TrainData_distribution_sorted = sorted(TrainData_distribution.items(), key = lambda d: d[1], reverse = True)
CVData_distribution_sorted = sorted(CVData_distribution.items(), key = lambda d: d[1], reverse = True)
TestData_distribution_sorted = sorted(TestData_distribution.items(), key = lambda d: d[1], reverse = True)


# In[365]:


fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Train Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_train)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.2, y = i.get_height()+1.5, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TrainData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_train.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Validation Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_cv)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in CVData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_cv.shape[0])*100), 4))+"%)")

print("-"*80)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_axes([0,0,1,1])
ax.set_title("Count of each Emotion in Test Data", fontsize = 20)
sns.countplot(x = "Emotion", data = df_temp_test)
plt.grid()
for i in ax.patches:
    ax.text(x = i.get_x() + 0.27, y = i.get_height()+0.2, s = str(i.get_height()), fontsize = 20, color = "grey")
plt.xlabel("")
plt.ylabel("Count", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()

for i in TestData_distribution_sorted:
    print("Number of training data points in class "+str(i[0])+" = "+str(i[1])+ "("+str(np.round(((i[1]/df_temp_test.shape[0])*100), 4))+"%)")


# ## 4. Pre-Processing Human Images

# ### 4.1 Converting all the images to grayscale and save them

# In[47]:


def convt_to_gray(df):
    count = 0
    for i in range(len(df)):
        path1 = df["folderName"][i]
        path2 = df["imageName"][i]
        img = cv2.imread(os.path.join(path1, path2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path1, path2), gray)
        count += 1
    print("Total number of images converted and saved = "+str(count))


# In[48]:


convt_to_gray(df_human_train)


# In[49]:


convt_to_gray(df_human_cv)


# In[50]:


convt_to_gray(df_human_test)


# ### 4.2 Detecting face in image using HAAR then crop it then resize then save the image

# In[94]:


#detect the face in image using HAAR cascade then crop it then resize it and finally save it.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
#download this xml file from link: https://github.com/opencv/opencv/tree/master/data/haarcascades.
def face_det_crop_resize(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]  #cropping the face in image
        cv2.imwrite(img_path, cv2.resize(face_clip, (299, 299)))  #resizing image then saving it

'''
My addition!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
def detectFaceOpenCVDnn( img_path):
    # import sys
    # sys.path.append('../')
    prototxt_path = os.path.join('./models/deploy.prototxt')
    caffemodel_path = os.path.join('./models/weights.caffemodel')

    # Read the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    # modelFile = "models/opencv_face_detector_uint8.pb"
    # configFile = "models/opencv_face_detector.pbtxt"
    # net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    img = cv2.imread(img_path)
    # frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    frameOpencvDnn = img.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 177, 123])

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    conf_threshold = 0.7
    # for i in range(detections.shape[2]):
    confidence = detections[0, 0, 0, 2]
    x1 = int(detections[0, 0, 0, 3] * frameWidth)
    y1 = int(detections[0, 0, 0, 4] * frameHeight)
    x2 = int(detections[0, 0, 0, 5] * frameWidth)
    y2 = int(detections[0, 0, 0, 6] * frameHeight)
        # bboxes.append([x1, y1, x2, y2])
        # cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

    # (x1,y1,x2,y2) = bboxes[0]
    if confidence > conf_threshold:
        face_clip = img[y1:y2, x1:x2]
        if face_clip != []:
            # print(img[y1:y2, x1:x2], img_path, x1,x2,y1,y2)
            print(img_path)
            cv2.imwrite(img_path, cv2.resize(face_clip, (299, 299)))
    # print("Image " + " converted successfully")

    # for (x1,y1,x2,y2) in bboxes:
    #     face_clip = img[y1:y2, x1:x2]  #cropping the face in image
    #     cv2.imwrite(img_path, cv2.resize(face_clip, (350, 350)))  #resizing image then saving it


'''
My addition!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# In[96]:


# for i, d in df_human_train.iterrows():
#     img_path = os.path.join(d["folderName"], d["imageName"])
#     face_det_crop_resize(img_path)
#     # detectFaceOpenCVDnn(img_path)


# # In[97]:


# for i, d in df_human_cv.iterrows():
#     img_path = os.path.join(d["folderName"], d["imageName"])
#     face_det_crop_resize(img_path)
#     # detectFaceOpenCVDnn(img_path)


# # In[98]:


# for i, d in df_human_test.iterrows():
#     img_path = os.path.join(d["folderName"], d["imageName"])
#     face_det_crop_resize(img_path)
#     # detectFaceOpenCVDnn(img_path)




def convt_to_gray(df):
    count = 0
    for i in range(len(df)):
        path1 = df["folderName"][i]
        path2 = df["imageName"][i]
        img = cv2.imread(os.path.join(path1, path2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path1, path2), gray)
        count += 1
    print("Total number of images converted and saved = "+str(count))



# ### 8.2 Crop the image then resize them then save them.

# In[330]:


def change_image(df):
    count = 0
    for i, d in df.iterrows():
        img = cv2.imread(os.path.join(d["folderName"], d["imageName"]))
        face_clip = img[40:240, 35:225]         #cropping the face in image
        face_resized = cv2.resize(face_clip, (299, 299))
        cv2.imwrite(os.path.join(d["folderName"], d["imageName"]), face_resized) #resizing and saving the image
        count += 1
    print("Total number of images cropped and resized = {}".format(count))



# ## 9. Combining train data of both Animated and Human images

# Remember, that here we have combined only the train images of both human and animated so that we can train our model on both human and animated images. However, we have kept CV and test images of both human and animated separate so that we can cross validation our results on both human and animated images separately. At the same time we will also be able to test the efficiency of our model separately on human and animated images. By this we will get to know that how well our model is performing on human and animated images separately.

# In[14]:


frames = [df_human_train]
combined_train = pd.concat(frames)
combined_train.shape


# In[16]:


combined_train = combined_train.sample(frac = 1.0)  #shuffling the dataframe
combined_train.reset_index(inplace = True, drop = True)
combined_train.to_pickle("../Data/Dataframes/combined_train.pkl")


# # ## 10. Creating bottleneck features from VGG-16 model. Here, we are using Transfer learning.

# # In[2]:


Train_Combined = pd.read_pickle("../Data/Dataframes/combined_train.pkl")
CV_Humans = pd.read_pickle("../Data/Dataframes/Human/df_human_cv.pkl")
# CV_Animated = pd.read_pickle("../Data/Dataframes/Animated/df_anime_cv.pkl")
Test_Humans = pd.read_pickle("../Data/Dataframes/Human/df_human_test.pkl")
# Test_Animated = pd.read_pickle("../Data/Dataframes/Animated/df_anime_test.pkl")

print(Train_Combined.shape, CV_Humans.shape, Test_Humans.shape)


# In[5]:


TrainCombined_batch_pointer = 0
CVHumans_batch_pointer = 0
CVAnimated_batch_pointer = 0
TestHumans_batch_pointer = 0
TestAnimated_batch_pointer = 0


# ### 10.1 Bottleneck features for CombinedTrain Data

# In[4]:


TrainCombined_Labels = pd.get_dummies(Train_Combined["Labels"]).as_matrix()
TrainCombined_Labels.shape


# In[5]:


def loadCombinedTrainBatch(batch_size):
    global TrainCombined_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = Train_Combined.iloc[TrainCombined_batch_pointer + i]["folderName"]
        path2 = Train_Combined.iloc[TrainCombined_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(TrainCombined_Labels[TrainCombined_batch_pointer + i]) #appending corresponding labels
        
    TrainCombined_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)


# In[1]:


#creating bottleneck features for train data using VGG-16- Image-net model
# model = InceptionResNetV2(weights='imagenet', include_top=False)
# SAVEDIR = "../Data/Bottleneck_Features/Bottleneck_CombinedTrain/"
# SAVEDIR_LABELS = "../Data/Bottleneck_Features/CombinedTrain_Labels/"
# batch_size = 10
# for i in range(int(len(Train_Combined)/batch_size)):
#     x, y = loadCombinedTrainBatch(batch_size)
#     print("Batch {} loaded".format(i+1))
    
#     np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
#     print("Creating bottleneck features for batch {}". format(i+1))
#     bottleneck_features = model.predict(x)
#     np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
#     print("Bottleneck features for batch {} created and saved\n".format(i+1))


# ### 10.2 Bottleneck features for CV Human

# In[57]:


CVHumans_Labels = pd.get_dummies(CV_Humans["Labels"]).as_matrix()
CVHumans_Labels.shape


# In[58]:


def loadCVHumanBatch(batch_size):
    global CVHumans_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = CV_Humans.iloc[CVHumans_batch_pointer + i]["folderName"]
        path2 = CV_Humans.iloc[CVHumans_batch_pointer + i]["imageName"]
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(CVHumans_Labels[CVHumans_batch_pointer + i]) #appending corresponding labels
        
    CVHumans_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)


# In[2]:


#creating bottleneck features for CV Human data using VGG-16- Image-net model
# model = InceptionResNetV2(weights='imagenet', include_top=False)
# SAVEDIR = "../Data/Bottleneck_Features/Bottleneck_CVHumans/"
# SAVEDIR_LABELS = "../Data/Bottleneck_Features/CVHumans_Labels/"
# batch_size = 10
# for i in range(int(len(CV_Humans)/batch_size)):
#     x, y = loadCVHumanBatch(batch_size)
#     print("Batch {} loaded".format(i+1))
    
#     np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
#     print("Creating bottleneck features for batch {}". format(i+1))
#     bottleneck_features = model.predict(x)
#     np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
#     print("Bottleneck features for batch {} created and saved\n".format(i+1))


# # ### 10.3 Bottleneck features for CV Animated

# # In[63]:


# CVAnimated_Labels = pd.get_dummies(CV_Animated["Labels"]).as_matrix()
# CVAnimated_Labels.shape


# # In[64]:


# def loadCVAnimatedBatch(batch_size):
#     global CVAnimated_batch_pointer
#     batch_images = []
#     batch_labels = []
#     for i in range(batch_size):
#         path1 = CV_Animated.iloc[CVAnimated_batch_pointer + i]["folderName"]
#         path2 = CV_Animated.iloc[CVAnimated_batch_pointer + i]["imageName"]
#         read_image = cv2.imread(os.path.join(path1, path2))
#         read_image_final = read_image/255.0  #here, we are normalizing the images
#         batch_images.append(read_image_final)
        
#         batch_labels.append(CVAnimated_Labels[CVAnimated_batch_pointer + i]) #appending corresponding labels
        
#     CVAnimated_batch_pointer += batch_size
        
#     return np.array(batch_images), np.array(batch_labels)


# # In[3]:


# #creating bottleneck features for CV Animated data using VGG-16- Image-net model
# model = VGG16(weights='imagenet', include_top=False)
# SAVEDIR = "../Data/Bottleneck_Features/Bottleneck_CVAnimated/"
# SAVEDIR_LABELS = "../Data/Bottleneck_Features/CVAnimated_Labels/"
# batch_size = 10
# for i in range(int(len(CV_Animated)/batch_size)):
#     x, y = loadCVAnimatedBatch(batch_size)
#     print("Batch {} loaded".format(i+1))
    
#     np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
#     print("Creating bottleneck features for batch {}". format(i+1))
#     bottleneck_features = model.predict(x)
#     np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
#     print("Bottleneck features for batch {} created and saved\n".format(i+1))


# ### 10.4 Bottleneck Features for Test Human Data

# In[66]:


TestHuman_Labels = pd.get_dummies(Test_Humans["Labels"]).as_matrix()
TestHuman_Labels.shape


# In[67]:


def loadTestHumansBatch(batch_size):
    global TestHumans_batch_pointer
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        path1 = Test_Humans.iloc[TestHumans_batch_pointer + i]["folderName"]
        path2 = Test_Humans.iloc[TestHumans_batch_pointer + i]["imageName"]
        
        read_image = cv2.imread(os.path.join(path1, path2))
        read_image_final = read_image/255.0  #here, we are normalizing the images
        batch_images.append(read_image_final)
        
        batch_labels.append(TestHuman_Labels[TestHumans_batch_pointer + i]) #appending corresponding labels
        
    TestHumans_batch_pointer += batch_size
        
    return np.array(batch_images), np.array(batch_labels)


# In[4]:


#creating bottleneck features for Test Humans data using VGG-16- Image-net model
# model = InceptionResNetV2(weights='imagenet',include_top=False) #(weights=None, input_shape=(350,350,3), classes=7, include_top=True)
# SAVEDIR = "../Data/Bottleneck_Features/Bottleneck_TestHumans/"
# SAVEDIR_LABELS = "../Data/Bottleneck_Features/TestHumans_Labels/"
# batch_size = 10
# for i in range(int(len(Test_Humans)/batch_size)):
#     x, y = loadTestHumansBatch(batch_size)
#     print("Batch {} loaded".format(i+1))
    
#     np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
#     print("Creating bottleneck features for batch {}". format(i+1))
#     bottleneck_features = model.predict(x)
#     np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
#     print("Bottleneck features for batch {} created and saved\n".format(i+1))

# leftover_points = len(Test_Humans) - TestHumans_batch_pointer
# if leftover_points != 0:
#     x, y = loadTestHumansBatch(leftover_points)
#     np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(int(len(Test_Humans)/batch_size) + 1)), y)
#     bottleneck_features = model.predict(x)
#     np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(int(len(Test_Humans)/batch_size) + 1)), bottleneck_features)


# # ### 10.5 Bottleneck Features for Test Animated Data

# # In[6]:


# TestAnimated_Labels = pd.get_dummies(Test_Animated["Labels"]).as_matrix()
# TestAnimated_Labels.shape


# # In[7]:


# def loadTestAnimatedBatch(batch_size):
#     global TestAnimated_batch_pointer
#     batch_images = []
#     batch_labels = []
#     for i in range(batch_size):
#         path1 = Test_Animated.iloc[TestAnimated_batch_pointer + i]["folderName"]
#         path2 = Test_Animated.iloc[TestAnimated_batch_pointer + i]["imageName"]
#         read_image = cv2.imread(os.path.join(path1, path2))
#         read_image_final = read_image/255.0  #here, we are normalizing the images
#         batch_images.append(read_image_final)
        
#         batch_labels.append(TestAnimated_Labels[TestAnimated_batch_pointer + i]) #appending corresponding labels
        
#     TestAnimated_batch_pointer += batch_size
        
#     return np.array(batch_images), np.array(batch_labels)


# # In[5]:


# #creating bottleneck features for Test Animated data using VGG-16- Image-net model
# model = VGG16(weights='imagenet', include_top=False)
# SAVEDIR = "../Data/Bottleneck_Features/Bottleneck_TestAnimated/"
# SAVEDIR_LABELS = "../Data/Bottleneck_Features/TestAnimated_Labels/"
# batch_size = 10
# for i in range(int(len(Test_Animated)/batch_size)):
#     x, y = loadTestAnimatedBatch(batch_size)
#     print("Batch {} loaded".format(i+1))
    
#     np.save(os.path.join(SAVEDIR_LABELS, "bottleneck_labels_{}".format(i+1)), y)
    
#     print("Creating bottleneck features for batch {}". format(i+1))
#     bottleneck_features = model.predict(x)
#     np.save(os.path.join(SAVEDIR, "bottleneck_{}".format(i+1)), bottleneck_features)
#     print("Bottleneck features for batch {} created and saved\n".format(i+1))


# ## 11. Modelling & Training

# In[3]:


no_of_classes = 7


# In[30]:


#model architecture
def model(input_shape):
    model = Sequential()
        
    model.add(Dense(512, activation='relu', input_dim = input_shape))
    model.add(Dropout(0.1))
    
    model.add(Dense(256, activation='relu'))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim = no_of_classes, activation='softmax')) 
    
    return model


# In[6]:


#training the model
SAVEDIR_COMB_TRAIN = "../Data/Bottleneck_Features/Bottleneck_CombinedTrain/"
SAVEDIR_COMB_TRAIN_LABELS = "../Data/Bottleneck_Features/CombinedTrain_Labels/"

SAVEDIR_CV_HUMANS = "../Data/Bottleneck_Features/Bottleneck_CVHumans/"
SAVEDIR_CV_HUMANS_LABELS = "../Data/Bottleneck_Features/CVHumans_Labels/"

# SAVEDIR_CV_ANIME = "../Data/Bottleneck_Features/Bottleneck_CVAnimated/"
# SAVEDIR_CV_ANIME_LABELS =  "../Data/Bottleneck_Features/CVAnimated_Labels/"

SAVER = "../Data/Model_Save/"

input_shape = 8*8*1536   #this is the shape of bottleneck feature of each image which comes after passing the image through VGG-16

model = model(input_shape) #InceptionResNetV2(weights=None, input_shape=(350,350,3), classes=7, include_top=True)
# model.load_weights(os.path.join(SAVER, "model.h5"))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

epochs = 20
batch_size = 10
step = 0
combTrain_bottleneck_files = int(len(Train_Combined) / batch_size)
CVHuman_bottleneck_files = int(len(CV_Humans) / batch_size)
# CVAnime_bottleneck_files = int(len(CV_Animated) / batch_size)
epoch_number, CombTrain_loss, CombTrain_acc, CVHuman_loss, CVHuman_acc, CVAnime_loss, CVAnime_acc = [], [], [], [], [], [], []

for epoch in range(epochs):
    avg_epoch_CombTr_loss, avg_epoch_CombTr_acc, avg_epoch_CVHum_loss, avg_epoch_CVHum_acc, avg_epoch_CVAnime_loss, avg_epoch_CVAnime_acc = 0, 0, 0, 0, 0, 0
    epoch_number.append(epoch + 1)
    
    for i in range(combTrain_bottleneck_files):
        
        step += 1
        
        #loading batch of train bottleneck features for training MLP.
        X_CombTrain_load = np.load(os.path.join(SAVEDIR_COMB_TRAIN, "bottleneck_{}.npy".format(i+1)))
        X_CombTrain = X_CombTrain_load.reshape(X_CombTrain_load.shape[0], X_CombTrain_load.shape[1]*X_CombTrain_load.shape[2]*X_CombTrain_load.shape[3])
        Y_CombTrain = np.load(os.path.join(SAVEDIR_COMB_TRAIN_LABELS, "bottleneck_labels_{}.npy".format(i+1)))
        
        #loading batch of Human CV bottleneck features for cross-validation.
        X_CVHuman_load = np.load(os.path.join(SAVEDIR_CV_HUMANS, "bottleneck_{}.npy".format((i % CVHuman_bottleneck_files) + 1)))
        X_CVHuman = X_CVHuman_load.reshape(X_CVHuman_load.shape[0], X_CVHuman_load.shape[1]*X_CVHuman_load.shape[2]*X_CVHuman_load.shape[3])
        Y_CVHuman = np.load(os.path.join(SAVEDIR_CV_HUMANS_LABELS, "bottleneck_labels_{}.npy".format((i % CVHuman_bottleneck_files) + 1)))
        
        #loading batch of animated CV bottleneck features for cross-validation.
        # X_CVAnime_load = np.load(os.path.join(SAVEDIR_CV_ANIME, "bottleneck_{}.npy".format((i % CVAnime_bottleneck_files) + 1)))
        # X_CVAnime = X_CVAnime_load.reshape(X_CVAnime_load.shape[0], X_CVAnime_load.shape[1]*X_CVAnime_load.shape[2]*X_CVAnime_load.shape[3])
        # Y_CVAnime = np.load(os.path.join(SAVEDIR_CV_ANIME_LABELS, "bottleneck_labels_{}.npy".format((i % CVAnime_bottleneck_files) + 1)))
        
        CombTrain_Loss, CombTrain_Accuracy = model.train_on_batch(X_CombTrain, Y_CombTrain) #train the model on batch
        CVHuman_Loss, CVHuman_Accuracy = model.test_on_batch(X_CVHuman, Y_CVHuman) #cross validate the model on CV Human batch
        # CVAnime_Loss, CVAnime_Accuracy = model.test_on_batch(X_CVAnime, Y_CVAnime) #cross validate the model on CV Animated batch
        
        print("Epoch: {}, Step: {}, CombTr_Loss: {}, CombTr_Acc: {}, CVHum_Loss: {}, CVHum_Acc: {}".format(epoch+1, step, np.round(float(CombTrain_Loss), 2), np.round(float(CombTrain_Accuracy), 2), np.round(float(CVHuman_Loss), 2), np.round(float(CVHuman_Accuracy), 2) ))
        
        avg_epoch_CombTr_loss += CombTrain_Loss / combTrain_bottleneck_files
        avg_epoch_CombTr_acc += CombTrain_Accuracy / combTrain_bottleneck_files
        avg_epoch_CVHum_loss += CVHuman_Loss / combTrain_bottleneck_files
        avg_epoch_CVHum_acc += CVHuman_Accuracy / combTrain_bottleneck_files
        # avg_epoch_CVAnime_loss += CVAnime_Loss / combTrain_bottleneck_files
        # avg_epoch_CVAnime_acc += CVAnime_Accuracy / combTrain_bottleneck_files
        
    print("Avg_CombTrain_Loss: {}, Avg_CombTrain_Acc: {}, Avg_CVHum_Loss: {}, Avg_CVHum_Acc: {}".format(np.round(float(avg_epoch_CombTr_loss), 2), np.round(float(avg_epoch_CombTr_acc), 2), np.round(float(avg_epoch_CVHum_loss), 2), np.round(float(avg_epoch_CVHum_acc), 2)))

    CombTrain_loss.append(avg_epoch_CombTr_loss)
    CombTrain_acc.append(avg_epoch_CombTr_acc)
    CVHuman_loss.append(avg_epoch_CVHum_loss)
    CVHuman_acc.append(avg_epoch_CVHum_acc)
    # CVAnime_loss.append(avg_epoch_CVAnime_loss)
    # CVAnime_acc.append(avg_epoch_CVAnime_acc)
    
    model.save(os.path.join(SAVER, "model.h5"))  #saving the model on each epoc
    model.save_weights(os.path.join(SAVER, "model_weights.h5")) #saving the weights of model on each epoch
    print("Model and weights saved at epoch {}".format(epoch + 1))
          
log_frame = pd.DataFrame(columns = ["Epoch", "Comb_Train_Loss", "Comb_Train_Accuracy", "CVHuman_Loss", "CVHuman_Accuracy"])
log_frame["Epoch"] = epoch_number
log_frame["Comb_Train_Loss"] = CombTrain_loss
log_frame["Comb_Train_Accuracy"] = CombTrain_acc
log_frame["CVHuman_Loss"] = CVHuman_loss
log_frame["CVHuman_Accuracy"] = CVHuman_acc
# log_frame["CVAnime_Loss"] = CVAnime_loss
# log_frame["CVAnime_Accuracy"] = CVAnime_acc
log_frame.to_csv("../Data/Logs/Log.csv", index = False)


# # In[40]:


log = pd.read_csv("../Data/Logs/Log.csv")
log


# In[41]:


def plotting(epoch, train_loss, CVHuman_loss, title):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train_loss, color = 'red', label = "Train")
    axes.plot(epoch, CVHuman_loss, color = 'blue', label = "CV_Human")
    # axes.plot(epoch, CVAnimated_loss, color = 'green', label = "CV_Animated")
    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel("Loss", fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)


# In[44]:


plotting(list(log["Epoch"]), list(log["Comb_Train_Loss"]), list(log["CVHuman_Loss"]), "EPOCH VS LOSS")


# In[47]:


def plotting(epoch, train_acc, CVHuman_acc, title):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train_acc, color = 'red', label = "Train_Accuracy")
    axes.plot(epoch, CVHuman_acc, color = 'blue', label = "CV_Human_Accuracy")
    # axes.plot(epoch, CVAnimated_acc, color = 'green', label = "CV_Animated_Accuracy")
    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel("Accuracy", fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)


# In[49]:


plotting(list(log["Epoch"]), list(log["Comb_Train_Accuracy"]), list(log["CVHuman_Accuracy"]), "EPOCH VS ACCURACY")


# ## 12. Checking Test Accuracy

# In[3]:


def print_confusionMatrix(Y_TestLabels, PredictedLabels):
    confusionMatx = confusion_matrix(Y_TestLabels, PredictedLabels)
    
    precision = confusionMatx/confusionMatx.sum(axis = 0)
    
    recall = (confusionMatx.T/confusionMatx.sum(axis = 1)).T
    
    sns.set(font_scale=1.5)
    
    # confusionMatx = [[1, 2],
    #                  [3, 4]]
    # confusionMatx.T = [[1, 3],
    #                   [2, 4]]
    # confusionMatx.sum(axis = 1)  axis=0 corresponds to columns and axis=1 corresponds to rows in two diamensional array
    # confusionMatx.sum(axix =1) = [[3, 7]]
    # (confusionMatx.T)/(confusionMatx.sum(axis=1)) = [[1/3, 3/7]
    #                                                  [2/3, 4/7]]

    # (confusionMatx.T)/(confusionMatx.sum(axis=1)).T = [[1/3, 2/3]
    #                                                    [3/7, 4/7]]
    # sum of row elements = 1
    
    labels = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]
    
    plt.figure(figsize=(16,7))
    sns.heatmap(confusionMatx, cmap = "Blues", annot = True, fmt = ".1f", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.show()
    
    print("-"*125)
    
    plt.figure(figsize=(16,7))
    sns.heatmap(precision, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
    plt.title("Precision Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.show()
    
    print("-"*125)
    
    plt.figure(figsize=(16,7))
    sns.heatmap(recall, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
    plt.title("Recall Matrix", fontsize = 30)
    plt.xlabel('Predicted Class', fontsize = 20)
    plt.ylabel('Original Class', fontsize = 20)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 90)
    plt.show()


# ### Test Data of Human Images

# In[4]:


model = load_model("../Data/Model_Save/model.h5")
predicted_labels = []
true_labels = []
batch_size = 10
if len(Test_Humans)%10 != 0:
    total_files = int(len(Test_Humans) / batch_size) + 2 #here, I have added 2 because there are 30 files in Test_Humans
else:
    total_files = int(len(Test_Humans) / batch_size) + 1 #here, I have added 2 because there are 30 files in Test_Humans
for i in range(1, total_files, 1):
    img_load = np.load("../Data/Bottleneck_Features/Bottleneck_TestHumans/bottleneck_{}.npy".format(i))
    img_label = np.load("../Data/Bottleneck_Features/TestHumans_Labels/bottleneck_labels_{}.npy".format(i))
    img_bundle = img_load.reshape(img_load.shape[0], img_load.shape[1]*img_load.shape[2]*img_load.shape[3])
    for j in range(img_bundle.shape[0]):
        img = img_bundle[j]
        img = img.reshape(1, img_bundle.shape[1])
        pred = model.predict(img)
        predicted_labels.append(pred[0].argmax())
        true_labels.append(img_label[j].argmax())
acc = accuracy_score(true_labels, predicted_labels)
print("Accuracy on Human Test Data = {}%".format(np.round(float(acc*100), 2)))


# In[5]:


print_confusionMatrix(true_labels, predicted_labels)


# ### Test Data of Animated Images

# In[6]:


# model = load_model("../Data/Model_Save/model.h5")
# predicted_labels = []
# true_labels = []
# batch_size = 10
# total_files = int(len(Test_Animated) / batch_size) + 1
# for i in range(1, total_files, 1):
#     img_load = np.load("../Data/Bottleneck_Features/Bottleneck_TestAnimated/bottleneck_{}.npy".format(i))
#     img_label = np.load("../Data/Bottleneck_Features/TestAnimated_Labels/bottleneck_labels_{}.npy".format(i))
#     img_bundle = img_load.reshape(img_load.shape[0], img_load.shape[1]*img_load.shape[2]*img_load.shape[3])
#     for j in range(img_bundle.shape[0]):
#         img = img_bundle[j]
#         img = img.reshape(1, img_bundle.shape[1])
#         pred = model.predict(img)
#         predicted_labels.append(pred[0].argmax())
#         true_labels.append(img_label[j].argmax())
# acc = accuracy_score(true_labels, predicted_labels)
# print("Accuracy on Animated Test Data = {}%".format(np.round(float(acc*100), 2)))


# # In[7]:


# print_confusionMatrix(true_labels, predicted_labels)


# ## 13. Testing on Real World with Still Images

# In[8]:


# Now for testing the model on real world images we have to follow all of the same steps which we have done on our training, CV
# and test images. Like here we have to first pre-preocess our images then create its VGG-16 bottleneck features then pass those 
# bottleneck features through our own MLP model for prediction.
# Steps are as follows:
# 1. Read the image, convert it to grayscale and save it.
# 2. Read that grayscale saved image, the detect face in it using HAAR cascade.
# 3. Crop the image to the detected face and resize it to 350*350 and save the image.
# 4. Read that processed cropped-resized image, then reshape it and normalize it.
# 5. Then feed that image to VGG-16 and create bottleneck features of that image and then reshape it.
# 6. Then use our own model for final prediction of expression.


# In[2]:


EMOTION_DICT = {1:"ANGRY", 2:"DISGUST", 3:"FEAR", 4:"HAPPY", 5:"NEUTRAL", 6:"SAD", 7:"SURPRISE"}
model_VGG = InceptionResNetV2(weights='imagenet', include_top=False)
model_top = load_model("../Data/Model_Save/model.h5")


# In[3]:


def make_prediction(path):
    #converting image to gray scale and save it
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, gray)
    
    #detect face in image, crop it then resize it then save it
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face_clip = img[y:y+h, x:x+w]
        cv2.imwrite(path, cv2.resize(face_clip, (299, 299)))
    
    #read the processed image then make prediction and display the result
    read_image = cv2.imread(path)
    read_image = read_image.reshape(1, read_image.shape[0], read_image.shape[1], read_image.shape[2])
    read_image_final = read_image/255.0  #normalizing the image
    VGG_Pred = model_VGG.predict(read_image_final)  #creating bottleneck features of image using VGG-16.
    VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1]*VGG_Pred.shape[2]*VGG_Pred.shape[3])
    top_pred = model_top.predict(VGG_Pred)  #making prediction from our own model.
    emotion_label = top_pred[0].argmax() + 1
    print("Predicted Expression Probabilities")
    print("ANGRY: {}\nDISGUST: {}\nFEAR: {}\nHAPPY: {}\nNEUTRAL: {}\nSAD: {}\nSURPRISE: {}\n\n".format(top_pred[0][0], top_pred[0][1], top_pred[0][2], top_pred[0][3], top_pred[0][4], top_pred[0][5], top_pred[0][6]))
    print("Dominant Probability = "+str(EMOTION_DICT[emotion_label])+": "+str(max(top_pred[0])))


# ### ANGRY

# ### Correct Result

# In[17]:


Image.open("../Data/Test_Images/Angry_1.JPG")


# In[20]:


make_prediction("../Data/Test_Images/Angry_1.JPG")


# ### Correct Result

# In[21]:


Image.open("../Data/Test_Images/Angry_2.png")


# In[22]:


make_prediction("../Data/Test_Images/Angry_2.png")


# ### DISGUST

# ### Incorrect Result

# In[21]:


Image.open("../Data/Test_Images/Disgust_1.jpg")


# In[20]:


make_prediction("../Data/Test_Images/Disgust_1.jpg")


# ### Correct Result

# In[31]:


Image.open("../Data/Test_Images/Disgust_2.png")


# In[32]:


make_prediction("../Data/Test_Images/Disgust_2.png")


# ### FEAR

# ### Correct Result

# In[17]:


Image.open("../Data/Test_Images/Fear_1.jpg")


# In[16]:


make_prediction("../Data/Test_Images/Fear_1.jpg")


# ### Correct Result

# In[6]:


Image.open("../Data/Test_Images/Fear_2.png")


# In[4]:


make_prediction("../Data/Test_Images/Fear_2.png")


# ### HAPPY

# ### Correct Result

# In[14]:


Image.open("../Data/Test_Images/Happy_1.jpg")


# In[23]:


make_prediction("../Data/Test_Images/Happy_1.jpg")


# ### Correct Result

# In[22]:


Image.open("../Data/Test_Images/Happy_2.png")


# In[24]:


make_prediction("../Data/Test_Images/Happy_2.png")


# ### Neutral

# ### Correct Result

# In[4]:


Image.open("../Data/Test_Images/Neutral_1.jpg")


# In[3]:


make_prediction("../Data/Test_Images/Neutral_6.jpg")


# ### Sad

# ### Correct Prediction

# In[9]:


Image.open("../Data/Test_Images/Sad_1.jpg")


# In[7]:


make_prediction("../Data/Test_Images/Sad_1.jpg")


# ### Correct Prediction

# In[11]:


Image.open("../Data/Test_Images/Sad_2.png")


# In[10]:


make_prediction("../Data/Test_Images/Sad_2.png")


# ### Surprise

# ### Correct Prediction

# In[13]:


Image.open("../Data/Test_Images/Surprise_1.jpg")


# In[12]:


make_prediction("../Data/Test_Images/Surprise_1.jpg")


# ### Correct Prediction

# In[16]:


Image.open("../Data/Test_Images/Surprise_2.png")


# In[15]:


make_prediction("../Data/Test_Images/Surprise_2.png")


# In[7]:


Image.open("../Data/Test_Images/Surprise_3.jpg")


# In[8]:


make_prediction("../Data/Test_Images/Surprise_3.jpg")


# In[337]:


# cnt_correct = 0
# cnt_incorrect = 0
# for i, d in df_anime_test.iterrows():
#     img_path = os.path.join(d["folderName"], d["imageName"])
#     im_size = cv2.imread(img_path).shape
#     if im_size == (350, 350, 3):
#         cnt_correct += 1
#     else:
#         cnt_incorrect += 1
# print("Correct = "+str(cnt_correct))
# print("incorrect = "+str(cnt_incorrect))


# In[123]:


# a = Train_Combined
# randInt = np.random.randint(0, a.shape[0], size = (1))[0]
# emotion = a["Emotion"][randInt]
# label = a["Labels"][randInt]
# path1 = a["folderName"][randInt]
# path2 = a["imageName"][randInt]
# img = Image.open(os.path.join(path1, path2))
# img


# In[124]:


# print(emotion)
# print(label)


# In[41]:


# count_present = 0
# count_absent = 0
# for i, d in df_angry_reduced.iterrows():
#     path1 = d["folderName"]
#     path2 = d["imageName"]
#     if os.path.isfile(os.path.join(path1, path2)):
#         count_present += 1
#     else:
#         count_absent += 1
# print("Count present = "+str(count_present))
# print("Count absent = "+str(count_absent))


# In[ ]:




