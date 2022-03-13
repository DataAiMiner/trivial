import pandas as pd
import numpy as np
data = pd.read_csv('./mnist_train_small.csv', header=None)
data = np.array(data)

y_train = data[:, 1:]
x_train = data[:,0]
y_train = y_train.reshape(-1, 28, 28)

import matplotlib.pyplot as plt
plt.imshow(y_train[1,:,:])
print(x_train[1])

import tensorflow.keras as keras

x_train = keras.utils.to_categorical(x_train)
y_train = y_train/255

from keras.models import Sequential
from keras import layers
from keras import optimizers

model = Sequential()
model.add(layers.Dense(units=3136, input_shape=(10,), activation='relu'))
model.add(layers.Reshape((7, 7, 64)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
model.complie(loss='mean_squared_error', optimizer=optimizers, RMSprop(learning_rate=0.001), metrics=['accuracy'])

model.build()
model.summary()

hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.3)
