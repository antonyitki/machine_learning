import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from glob import glob
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')
import time
# https://www.kaggle.com/code/officialsubhash/rock-paper-scissor-image-classification-99
# https://medium.com/analytics-vidhya/indian-food-image-classification-using-transfer-learning-b8878187ddd1


# record start time
start = time.time()
path = "rps-cv-images\\*\\*.png"
all_files = glob(path)
len(all_files)
images = []
labels = []
for path in all_files:
    img = load_img(path, target_size=(100,100))
    img = img_to_array(img, dtype=np.uint8)
    images.append(img)
    label = path.split("\\")[-2]
    labels.append(label)
images = np.array(images)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
images.shape, labels.shape
CLASSES = label_encoder.classes_.tolist()
CLASSES
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=1)
X_train.shape, y_train.shape
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
axs = np.ravel(axs)
for i in range(len(axs)):
    plt.sca(axs[i])
    plt.imshow(X_train[i])
    plt.axis('off')
plt.show()
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
y_train_oh = to_categorical(y_train, num_classes=3)
y_test_oh = to_categorical(y_test, num_classes=3)
tf.random.set_seed(10)
tf.random.set_seed(10)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', input_shape=X_train[0].shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', input_shape=X_train[0].shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', input_shape=X_train[0].shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', input_shape=X_train[0].shape),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, 'softmax')
])
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train_oh, batch_size=32, epochs=10, validation_data=(X_test_scaled, y_test_oh))
print(model.evaluate(X_test, y_test_oh))
predictions = model.predict(X_test_scaled).argmax(axis=1)
print(predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names = ['p', 'r', 's']))
cm2 = confusion_matrix(y_test, predictions)
df_cm = pd.DataFrame(cm2, index = [i for i in ['p', 'r', 's']],columns = [i for i in ['p', 'r', 's']])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,cmap="RdPu")
plt.show()
plt.savefig("confmatrixbis.png")
model.save("bis_program_model.h5")
# record end time
end = time.time()
print("The time of execution of above program is :",(end-start) , "s")