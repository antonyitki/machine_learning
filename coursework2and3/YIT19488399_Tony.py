###############################################################################
# 
# Coursework 2 & 3       Value=50%       2022.23 Machine Learning - CMP020X303A
# Module Tutor: Dr Gu, delivery date 23 of November 2022
#
# Creator, author and owner: Tony (YIT19488399), 
# BSc Computer Science, year 3
# Name of the program: "clasima"
#  
# Field program: ML, image classification with labels.     Neural Network (NN).
#
###############################################################################

# nhttps://www.geeksforgeeks.org/introduction-convolution-neural-network/
# https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import time

ACCURACY_SELECTION_TRAINING = 0.9555


# record start time
start = time.time()
src_dir = 'archive'
paper_dir = os.path.join(src_dir,'paper')
rock_dir = os.path.join(src_dir,'rock')
scissors_dir = os.path.join(src_dir,'scissors')
print("\nTotal number of images:")
print('paper : ',len(os.listdir(paper_dir)))
print('rock : ',len(os.listdir(rock_dir)))
print('scissors : ',len(os.listdir(scissors_dir)))
generator = ImageDataGenerator(validation_split=0.4, rescale=1/255, shear_range=0.2, zoom_range=0.2, rotation_range=20, fill_mode='nearest')
train_data = generator.flow_from_directory(src_dir, batch_size=32, target_size=(150,150), subset='training')
val_data = generator.flow_from_directory(src_dir, batch_size=32,  target_size=(150,150), subset='validation')
print(f"total number of images in the training: {train_data.samples}")
print(f"total number of images in the validation: {val_data.samples}")
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.summary()
print(model.to_json())


class Callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= ACCURACY_SELECTION_TRAINING):
            print("\nReached %2.2f%% accuracy, training has been stop" %(logs.get('accuracy')*100))
            self.model.stop_training = True
callbacks = Callbacks()


model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
historyModel = model.fit(train_data, steps_per_epoch = 25, epochs = 30, validation_data = val_data,
    validation_steps = 5, verbose = 2, callbacks = [callbacks])
print(model.history)
uploaded = os.path.join('WIN_20221029_18_54_09_Pro.jpg') # scissors
img = tf.keras.preprocessing.image.load_img(uploaded, target_size = (150, 150)) 
imgplot = plt.imshow(img)
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)  
print(classes)
if classes[0][0] == 1:
  print('Paper')
elif classes[0][1] == 1:
  print('Rock')
else:
  print('Scissors')
model.save("first_program_model.h5")
plt.subplot(1, 2, 1)
plt.plot(historyModel.history['loss'])
plt.plot(historyModel.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.subplot(1, 2, 2)
plt.plot(historyModel.history['accuracy'])
plt.plot(historyModel.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
predictions = model.predict(val_data)
print(predictions)
PARENT_DIR = 'archive'
rock_child_dir = os.path.join(PARENT_DIR,'rock')
paper_child_dir = os.path.join(PARENT_DIR,'paper')
scissors_child_dir = os.path.join(PARENT_DIR,'scissors')
os.listdir(PARENT_DIR)
rock_loaded_img = os.listdir(rock_child_dir)
paper_loaded_img = os.listdir(paper_child_dir)
scissors_loaded_img = os.listdir(scissors_child_dir)
plt.figure(figsize=(10, 4))
for i, img_path in enumerate(rock_loaded_img[:3]):
    sp = plt.subplot(1, 3, i+1)
    img = mpimg.imread(os.path.join(rock_child_dir, img_path))
    plt.imshow(img)
plt.show()
plt.figure(figsize=(10, 4))
for i, img_path in enumerate(paper_loaded_img[:3]):
    sp = plt.subplot(1, 3, i+1)
    img = mpimg.imread(os.path.join(paper_child_dir, img_path))
    plt.imshow(img)
plt.show()
plt.figure(figsize=(10, 4))
for i, img_path in enumerate(scissors_loaded_img[:3]):
    sp = plt.subplot(1, 3, i+1)
    img = mpimg.imread(os.path.join(scissors_child_dir, img_path))
    plt.imshow(img)
plt.show()
# record end time
end = time.time()
print("The time of execution of above program is :",(end-start) , "s")