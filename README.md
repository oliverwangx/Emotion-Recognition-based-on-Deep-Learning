# Emotion-Recognition-based-on-Deep-Learning


This is the undergraduate capstone project


### Problem Statement

Emotion Recognition, or specifically Facial Expression Recognition (FER) has been one of the most prevalent topics in the application of deep learning. As a powerful method of monitering human emotion, FER model can be unilized in an automotive in order to detecting driver's negative emotions and thus increase driving safety. Typically, a
FER model should be able to recognize seven emotions: happy, sad, angry, surprise, disgust, fear, and neutral.

### Design Description

The design uses Multi-task Convolutional Neural Network (MTCNN) to detect and crop face region from background. After the step, a homography function is employed to align all the faces based on the landmarks returned by MTCNN. Then the preprocessed data set is expanded by data augmentation methods, e.g. changing lightness and contrast, adding Gaussian flur or noise, etc. With the augmented data, a pre-trained model of Inception Residual Network V2 is further
trained and a final model is obtained.


![image](https://github.com/oliverwangx/Emotion-Recognition-based-on-Deep-Learning/blob/master/Team11_Poster%20(1).png)


