# Intro
This is result of my learning of Deep Learning basics. The code is based on and in some of its part copied from the following tutorial: https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/

## Step 1 - training data
Prepare training set. Normalize data size.


```python
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "pics_down/pics"
CATEGORIES = ["duda", "trzaskowski"]
training_data = []
IMG_SIZE = 100

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # classification: 0=duda 1=trzaskowksi

        for img in tqdm(os.listdir(path)):  # iterate over images and prepare a training set
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

```

    100%|██████████| 138/138 [00:00<00:00, 251.19it/s]
    100%|██████████| 235/235 [00:00<00:00, 242.46it/s]


Shuffle traning data - otherwise the classifier would learn to just predict first always Duda, then Trzaskowski.


```python
import random
random.shuffle(training_data)
```


```python
X = []  # data
y = []  # prediction target

for features,label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
```

Save training data


```python
import pickle

pickle_out = open("X_p.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_p.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
```

## Step 2 - Build model
Create a model.

Convolution - the act of taking the original data, and creating feature maps from it.

Pooling - down-sampling, most often in the form of "max-pooling," where we select a region, and then take the maximum value in that region, and that becomes the new value for the entire region.

Hidden layer = convolution + pooling


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X_p.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_p.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])

model.save('64x3-CNN_p.model')
```

    3-conv-64-nodes-0-dense-1595273969
    Train on 252 samples, validate on 109 samples
    Epoch 1/10
    252/252 [==============================] - 6s 25ms/sample - loss: 0.6979 - accuracy: 0.5397 - val_loss: 0.6408 - val_accuracy: 0.6606
    Epoch 2/10
    252/252 [==============================] - 4s 17ms/sample - loss: 0.6748 - accuracy: 0.6190 - val_loss: 0.6619 - val_accuracy: 0.6789
    Epoch 3/10
    252/252 [==============================] - 6s 23ms/sample - loss: 0.6666 - accuracy: 0.6151 - val_loss: 0.6204 - val_accuracy: 0.6606
    Epoch 4/10
    252/252 [==============================] - 6s 26ms/sample - loss: 0.6467 - accuracy: 0.6270 - val_loss: 0.6171 - val_accuracy: 0.6789
    Epoch 5/10
    252/252 [==============================] - 4s 17ms/sample - loss: 0.6154 - accuracy: 0.6786 - val_loss: 0.5824 - val_accuracy: 0.7339
    Epoch 6/10
    252/252 [==============================] - 4s 15ms/sample - loss: 0.5552 - accuracy: 0.7143 - val_loss: 0.5775 - val_accuracy: 0.7248
    Epoch 7/10
    252/252 [==============================] - 6s 25ms/sample - loss: 0.5105 - accuracy: 0.7619 - val_loss: 0.5663 - val_accuracy: 0.7248
    Epoch 8/10
    252/252 [==============================] - 5s 21ms/sample - loss: 0.4501 - accuracy: 0.7976 - val_loss: 0.6954 - val_accuracy: 0.6147
    Epoch 9/10
    252/252 [==============================] - 4s 16ms/sample - loss: 0.4665 - accuracy: 0.8016 - val_loss: 0.5502 - val_accuracy: 0.7615
    Epoch 10/10
    252/252 [==============================] - 7s 28ms/sample - loss: 0.4080 - accuracy: 0.8294 - val_loss: 0.5362 - val_accuracy: 0.7982
    WARNING:tensorflow:From /Users/piotrbajsarowicz/.pyenv/versions/learning_python/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    INFO:tensorflow:Assets written to: 64x3-CNN_p.model/assets


## Step 3 - Predict
Use the model and try to predict


```python
import tensorflow as tf

CATEGORIES = ["duda", "trzaskowski"]


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN_p.model")

for name in ["trzask", "duda"]:
    for item in range(1, 10):
        try:
            prediction = model.predict([prepare(f'test/{name}{item}.jpg')])
        except:
            continue
        result = CATEGORIES[int(prediction[0][0])]
        print(f"{name}{item} = {result}")
```

    trzask1 = = duda
    trzask2 = trzaskowski
    trzask3 = trzaskowski
    trzask4 = duda
    duda1 = trzaskowski
    duda2 = duda
    duda3 = duda

