# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:29:19 2022

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential





data_dir = pathlib.Path('./22 E proba M/types recepteurs')

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

batch_size = 32
img_height = 40
img_width = 40

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")




AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#for image_batch, labels_batch in train_ds:
 # print(image_batch.shape)
  #print(labels_batch.shape)
  #break
  
normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.summary()


epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



img = tf.keras.utils.load_img(
    '../22 E proba M/Lower/x164y906r12.png', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)





def getLabels(trainedModel,nameList,color='grayscale'):
    ll = []
    for nn in nameList:
        nnn = data_dir+"/"+nn
        ii = tf.keras.utils.load_img(nnn,color_mode=color,target_size=(img_width,img_height))
        ll.append(ii)
    label_List = []
    for im in ll:
         tt = tf.keras.preprocessing.image.img_to_array(im)
         tt = tf.expand_dims(tt, 0)
         rr = trainedModel.predict(tt)
         ind = rr[0].argmax()
         label_List.append(ind)
    return label_List


#data_augmentation = tf.keras.Sequential(
#    [
#        tf.keras.layers.RandomFlip("horizontal"),
#        tf.keras.layers.RandomRotation(0.1),
#    ]
#)


class_num_training_samples = {}
for f in train_ds.file_paths:
    class_name = f.split('/')[-2]
    if class_name in class_num_training_samples:
        class_num_training_samples[class_name] += 1
    else:
        class_num_training_samples[class_name] = 1
max_class_samples = max(class_num_training_samples.values())
class_weights = {}
for i in range(0, len(train_ds.class_names)):
    class_weights[i] = max_class_samples/class_num_training_samples[train_ds.class_names[i]]
   
    
def learnClasses1(train_ds,val_ds,nEpochs=32,cw = {}):
  
   earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 0,patience = 20, verbose = 1,restore_best_weights = True)
   
  
   inputs = tf.keras.Input(shape = (img_height,img_width,1))
   x = data_augmentation(inputs)
   x = tf.keras.layers.Rescaling(1.0 / 255)(x)
   x= tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
   x = tf.keras.layers.MaxPooling2D()(x)
   x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
   x = tf.keras.layers.MaxPooling2D()(x)
   x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
   x =  tf.keras.layers.MaxPooling2D()(x)
   x = tf.keras.layers.Flatten()(x)
   x = tf.keras.layers.Dense(128, activation='relu')(x)
   outputs = tf.keras.layers.Dense(num_classes,activation="softmax")(x)
  
   model = tf.keras.Model(inputs=inputs,outputs = outputs)
  
   optimizer = 'adam'
  
   # model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
   model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
   model.summary()
   history = model.fit(train_ds, epochs=nEpochs, validation_data=val_ds, verbose=1,class_weight=cw,callbacks=[earlystop])

   hd = history.history
   lv = hd['loss']
   lv1 = hd['val_loss']
   acc = hd['accuracy']
   acc1 = hd['val_accuracy']
   epochs = range(1,len(lv)+1)
   plt.figure(1)
   plt.clf()
   plt.subplot(2,1,2)
   plt.plot(epochs,acc,'o-')
   plt.plot(epochs,acc1,'o-')
   plt.subplot(2,1,1)
   plt.plot(epochs,lv,'o-')
   plt.plot(epochs,lv1,'o-')
   return model


