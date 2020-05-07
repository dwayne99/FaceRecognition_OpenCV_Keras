## This script is for the training process 

# Necessary packages
from keras.layers import Lambda, Input, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# resize all images
image_size = [224,224]

# path to train and test folders
train_path = 'data/train'
val_path = 'data/test'

# add preprocessing layer to the from of VGG
vgg = VGG16(
    # 3 for the RGB channel
    input_shape = image_size + [3],
    weights = 'imagenet',
    include_top = False)

# Don't train starting layers
for layer in vgg.layers:
  layer.trainable = False

# useful for getting number of classes
folders = glob('data/train/*')
print(f'Number of classes: {len(folders)}')

# Custom layers

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = 'softmax')(x)

# Model

model = Model(inputs = vgg.input, output = prediction)

print(model.summary())

# build the model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

# Setting up the Datagenrators
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale=1/255)

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
    val_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')

# Training the model 
history = model.fit_generator(
    training_set,
    validation_data = test_set,
    epochs = 5,
)

# Viewing the loss and accuracy

plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label = 'val accuracy')
plt.legend()
plt.show()

model.save('vgg_model1.h5')
