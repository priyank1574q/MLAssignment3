
# coding: utf-8

# In[1]:


# module load pythonpackages/3.6.0
# module load pythonpackages/3.6.0/numpy/1.15.0/gnu
# module load pythonpackages/3.6.0/pandas/0.23.4/gnu
# module load apps/pythonpackages/3.6.0/tensorflow/1.9.0/gpu
# module load apps/pythonpackages/3.6.0/keras/2.2.2/gpu
# module load lib/cudnn/7.0.3/precompiled


# In[2]:


import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D 
from keras.utils import np_utils
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD, Nadam
from keras import initializers
from keras.metrics import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th')

import os
import sys


# In[4]:

x_train_path = sys.argv[1]
train_path = os.path.abspath(x_train_path)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

x_train = (pd.read_csv(x_train_path, header = None, na_filter = False, low_memory = False)).values

y_train = np.array(x_train[:,0], dtype = np.int8)
y_train = np_utils.to_categorical(y_train)

x_train = np.array(x_train[:,1:], dtype = np.float32)
# x_train = scaler.fit_transform(x_train)
x_train = x_train/255.
x_train = x_train.reshape(x_train.shape[0], 1, 32, 32)


# In[5]:

x_test_path = sys.argv[2]
test_path = os.path.abspath(x_test_path)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_test = (pd.read_csv(x_test_path, header = None, na_filter = False, low_memory = False)).values

x_test = np.array(x_test[:,1:], dtype = np.float32)
# x_test = scaler.fit_transform(x_test)
x_test = x_test/255.
x_test = x_test.reshape(x_test.shape[0], 1, 32, 32)


# In[6]:


classes = y_train.shape[1]


# In[11]:

aug = ImageDataGenerator(zoom_range=0.1, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
aug.fit(x_train)

# In[12]:


model = Sequential([
    Conv2D(256, (5, 5), strides=1, input_shape=(1, 32, 32), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(128, (5, 5), strides=1, activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(64, (5, 5), strides=1, activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(1024, activation = 'tanh', kernel_initializer = keras.initializers.glorot_normal(seed=0)),
    Dense(512, activation = 'tanh', kernel_initializer = keras.initializers.glorot_normal(seed=0)),
    Dense(classes, activation = 'softmax', kernel_initializer = keras.initializers.glorot_normal(seed=0))
])

sgd = SGD(lr = 0.001)
adam = Adam(lr=0.001)
nadam = Nadam(lr=0.001)

model.compile(optimizer=adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor='val_acc', mode = 'max', patience=12, verbose=0)
model_checkpoint = ModelCheckpoint("best1.model", monitor='val_acc', mode = 'max', save_best_only=True, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', mode = 'max', factor=0.5, patience=4, min_lr=0.00001, verbose=0)

# In[13]:

model.fit_generator(aug.flow(x_train, y_train, batch_size=100), steps_per_epoch=x_train.shape[0]/100, epochs=2, verbose=0)

# In[14]:

model.fit(x_train, y_train, validation_split=0.1, epochs=100, shuffle=True, batch_size=100, callbacks=[early_stopping, model_checkpoint, reduce_lr], verbose=0)


# In[15]:


predictions = model.predict_classes(x_test, batch_size=10, verbose=0)


# In[16]:


x_output = sys.argv[3]
actual_path_out = os.path.abspath(x_output)
path_out = os.path.dirname(actual_path_out)
os.chdir(path_out)
np.savetxt(x_output, predictions)
