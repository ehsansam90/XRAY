#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model


# In[8]:


pic = load_img("NOT_NORMAL\person1_bacteria_1.jpeg")


# In[10]:


pic_array = img_to_array(pic)


# In[11]:


pic_array


# In[ ]:




