# continue from rnn_and_param_optimizer.py
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential

model = Sequential()
model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(8, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=2))
model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=2, input_dim=2))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3, strides=2, input_dim=2))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.3))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

hist = model.fit(x=x_train, y=y_train_cat, epochs=80, validation_split=0.3, batch_size=16)
o = model.predict(x_test)
o = np.argmax(p, 1)
sum(np.equal(y_test, o))/len(y_test)
