nTrain = 3000
nTest = 1000
x_train = cp.concatenate((data_no[:nTrain,:], data_ab[:nTrain,:]),0)
y_train = np.concatenate((np.zeros(nTrain,), np.ones(nTrain,)),0)
x_test = np.concatenate((data_no[nTrain:nTrain+nTest,:], data_ab[nTrain:nTrain+nTrst,:]),0)
y_test = np.concatenate((np.zeros(nTest,), np.ones(nTest,)),0)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizres
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = Sequential()
model.add(layers.Conv1D(filters=16, kernel_size=3, input_shape=(x_train.shape[1], 1), activation='relu'))
model.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=3, strides=2))
model.add(layers.Conv1D(filters=32, kernel_size=3, input_shape=(x_train.shape[1], 1), activvation='relu'))
model.add(layers.MaxPooling1D(pool_size=3, strides=2))
model.add(layers.LSTM(16))
model.add(layers.Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

model.fit(s_train, y_train, epochs=50, batch_size=128, validation_split=0.2)

o = model.predict(x_test)
o = np.argmax(o, 1)
y_test = np.argmax(y_test, 1)
sum(np.equal(y_test,o))/len(y_test)
