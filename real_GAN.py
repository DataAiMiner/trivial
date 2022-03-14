import pandas as pd
import numpy as np
import tensorflow.keras as keras
from keras import Sequential
from keras import layers, optimizers

d = pd.read_csv('./mnist_test.csv', header=None)
d = np.array(0)
y_train = d[:,1:].reshape(-1, 28, 28)
y_train = y_train/255

model_gen = Sequential()
model_gen.add(layers.Dense(units=3136, activation='relu'))
model_gen.add(layers.BatchNormalization())
model_gen.add(layers.Reshape((7, 7, 64)))
model_gen.add(layers.UpSampling2D((2, 2)))
model_gen.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_gen.add(layers.BatchNormalization())
model_gen.add(layers.UpSampling2D((2, 2)))
model_gen.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_gen.add(layers.BatchNormalization())
model_gen.add(layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
model_gen.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.002), metrics=['accuracy'])

model_disc = Sequential()
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', input_shape=(28, 28, 1), activation='relu'))
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_disc.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_disc.add(layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model_disc.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
model_disc.add(layers.Flatten())
model_disc.add(layers.Dense(units=1, activation='sigmoid'))
model_disc.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

model_comb = Sequential()
model_comb.add(model_gen)
model_comb.add(model_disc)
model_disc.trainable = False
model_comb.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

nOrig = 64
nGen = nOrig
vector_size = 10
nEpoch = 2000

for i in range(nEpoch):
  y_gen = np.zeros((nGen, 28, 28))
  test_input = np.random.rand(nGen, vector_size)
  for j in range(nGen):
    o = model_gen.predict(test_input[j,:].reshape(1, 10))
    o = o.reshape((28, 28))
    y_gen[j, :] = o
  y_gen = np.expand_dims(y_gen, -1)
  
  idx = np.array(range(y_train.shape[0]))
  np.random.shuffle(idx)
  idx = idx[:nOrig]
  y_orig = y_train[idx, :, :]
  y_orig = np.expand_dims(y_orig, -1)
  
  test_img = np.concatenate((y_gen, y_orig), 0)
  test_target = np.concatenate((np.zeros(y_gen.shape[0]), np.ones(y_gen.shape[0])), 0)
  
  loss_disc = model.disc.train_on_batch(test_img, test_target)
  
  loss_gen = model_comb.train_on_batch(test_input, np.ones(test_input.shape[0]))

  fig = plt.figure(figsize=[20, 10])
  test_input = np.random.rand(nGen, vector_size)
  for i in rage(nGen):
    o = model_gen.predict(test_input[i, :].reshape(1, vector_size))
    o = o.reshape((28, 28))
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(o)
