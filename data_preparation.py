#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import os
import random

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras import utils


# In[2]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


# In[3]:


not_normal = load_images_from_folder("NOT_NORMAL")
normal = load_images_from_folder("NORMAL")


# In[4]:


plt.imshow(normal[0])


# In[5]:


print(str(len(not_normal)/len(normal)) + " Is the rate of two categories ")


# In[34]:


#Data has imblance issue, 
#applying oversampeling method on not_normal 


# In[6]:


not_normal[0].shape


# In[7]:


generator = ImageDataGenerator(rotation_range=10 , brightness_range=[0.6,1.4], zoom_range=[0.8,1.2], horizontal_flip=True, height_shift_range=None,
                               width_shift_range=None)


# In[39]:


not_normal[0].shape


# In[40]:


img_2 = np.expand_dims(not_normal[0],0)


# In[41]:


plt.imshow(img_2[0])


# In[42]:


img_2.shape


# In[59]:


aug_iter = generator.flow(img_2)


# In[61]:


aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]


# In[62]:


len(aug_images)


# In[45]:


for i in aug_images:
    plt.imshow(i)


# In[46]:


fig, axes = plt.subplots(1,10, figsize=(20,5))
for img,ax in zip(aug_images, axes):
    ax.imshow(img)


# In[69]:


def aug_all(list):
    all_images =[]
    for img in list:
        aug_iter = generator.flow(np.expand_dims(img,0))
        #generate 10 of each image
        aug_image = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
        all_images.extend(aug_images)
    return(all_images)        


# In[70]:


test_list = [not_normal[0],not_normal[1]]


# In[71]:


test_aug = aug_all(test_list)


# In[76]:


not_normal_aug = aug_all(not_normal)


# In[77]:


len(not_normal_aug)


# In[78]:


not_normal_aug[0].shape


# In[82]:


y_train_binary = utils.to_categorical(y_train, num_classes)


# In[132]:


train_X = normal + not_normal_aug


# In[131]:


len(train)


# In[127]:


len(normal), len(not_normal_aug)


# In[134]:


train_Y = np.concatenate((np.zeros(660),np.ones(600)))


# In[154]:


dataset = pd.DataFrame({'image': train_X, 'normal_or_not': train_Y})


# In[155]:


#shuffeling dataframe
final_df = dataset.sample(frac=1)


# In[159]:


final_df.to_csv("labeled_data.csv")


# In[ ]:





# In[ ]:




