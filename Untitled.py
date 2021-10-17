#!/usr/bin/env python
# coding: utf-8

# In[43]:


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


# In[21]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


# In[18]:


not_normal = load_images_from_folder("NOT_NORMAL")
normal = load_images_from_folder("NORMAL")


# In[24]:


plt.imshow(normal[0])


# In[33]:


print(str(len(not_normal)/len(normal)) + " Is the rate of two categories ")


# In[34]:


#Data has imblance issue, 
#applying oversampeling method on not_normal 


# In[37]:


not_normal[0].shape


# In[73]:


generator = ImageDataGenerator(rotation_range=10 , brightness_range=[0.6,1.4], zoom_range=[0.8,1.2], horizontal_flip=True, height_shift_range=None,
                               width_shift_range=None, data_format=())


# In[59]:


test_image = random.choice(os.listdir("NOT_NORMAL"))
image_path = "NOT_NORMAL/" + test_image


# In[60]:


image_path


# In[68]:


img = np.expand_dims(plt.imread(image_path),0)
plt.imshow(img[0])


# In[76]:


img.shape


# In[88]:


aug_iter = generator.flow(img.reshape(1,1310,1646,1))


# In[99]:


aug_images = [next(aug_iter)[0].astype(np.uint8).reshape(1310,1646) for i in range(10)]


# In[106]:


for i in aug_images:
    plt.imshow(i)


# In[114]:


fig, axes = plt.subplots(1,10, figsize=(20,5))
for img,ax in zip(aug_images, axes):
    ax.imshow(img)


# In[ ]:




