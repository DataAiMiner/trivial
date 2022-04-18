import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model
import warnings
warnings.filterwarnings("ignore")


# Import Dataset - 5분 단위
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/all_metric_openapi_serve_prod.csv')
df.reset_index(inplace=True)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df.drop('index', inplace=True, axis=1)
print(len(df))
print(df.head(3))

# 결측치 처리
df = df.replace('NaN', np.nan)
df = df.replace('', np.nan)
df = df.replace(' ', np.nan)
df = df.fillna(method='ffill')  # 그냥 df.fillna(method='ffill') 만 하고 df에 재할당 하지 않으면 적용이 안 된다.

# 데이터셋 쪼개기
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

print(len(train_df))
print(len(val_df))
print(len(test_df))

# 데이터셋별 시간 범위 확인
train_df_idx = train_df.index
print('Min datetime from TRAIN set: %s' % train_df_idx.min())
print('Max datetime from TRAIN set: %s' % train_df_idx.max())
val_df_idx = val_df.index
print('Min datetime from VALIDATION set: %s' % val_df_idx.min())
print('Max datetime from VALIDATION set: %s' % val_df_idx.max())
test_df_idx = test_df.index
print('Min datetime from TEST set: %s' % test_df_idx.min())
print('Max datetime from TEST set: %s' % test_df_idx.max())

# # Scaling Dataset
# train = train_df
# scalers={}
# for i in train_df.columns:
#     scaler = MinMaxScaler(feature_range=(0,1))
#     s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
#     s_s = np.reshape(s_s,len(s_s))
#     scalers['scaler_'+ i] = scaler
#     train[i]=s_s
# val = val_df
# for i in val_df.columns:
#     scaler = scalers['scaler_'+i]
#     s_s = scaler.transform(val[i].values.reshape(-1,1))
#     s_s = np.reshape(s_s,len(s_s))
#     scalers['scaler_'+i] = scaler
#     val[i] = s_s
# test = test_df
# for i in train_df.columns:
#     scaler = scalers['scaler_'+i]
#     s_s = scaler.transform(test[i].values.reshape(-1,1))
#     s_s = np.reshape(s_s,len(s_s))
#     scalers['scaler_'+i] = scaler
#     test[i] = s_s

# 생각해볼만한 부분 : https://datascience.stackexchange.com/questions/27628/sliding-window-leads-to-overfitting-in-lstm
def split_series(series, n_past, n_future):
  # n_past ==> no of past observations
  # n_future ==> no of future observations
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

# 지난 60분(12 step)의 관측치로 다음 10분(2 Step)의 관측치를 예측
n_past = 12
n_future = 2
n_features = 15
X_train, y_train = split_series(train.values, n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features)) # cannot reshape array of size 1130112 into shape (5886,12,15)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
X_val, y_val = split_series(val. values,n_past, n_future)
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], n_features))

# Mine : One Encoder - One Decoder
# n_features ==> no of features at each timestep in the data.
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(32, return_state=True)
encoder_outputs = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs[1:]
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs[0])
decoder_l1 = tf.keras.layers.LSTM(32, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
my_model = tf.keras.models.Model(encoder_inputs,decoder_outputs)
my_model.summary()


# One Encoder - One Decoder
# n_features ==> no of features at each timestep in the data.
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)
model_e1d1.summary()

# Two Encoder - Two Decoder
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
model_e2d2.summary()


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 0.1 * 0.90 ** x)
model_e1d1.compile(optimizer = tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e1d1 = model_e1d1.fit(X_train, y_train, epochs=25, validation_data=(X_val,y_val), batch_size=32, verbose=0, callbacks=[reduce_lr]) 
# Input 0 of layer "model_2" is incompatible with the layer: expected shape=(None, 843, 15), found shape=(None, 12, 15)
model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history_e2d2 = model_e2d2.fit(X_train, y_train ,epochs=25,validation_data=(X_val,y_val), batch_size=32, verbose=0, callbacks=[reduce_lr])

plt.plot(history_e1d1.history['loss'])
plt.plot(history_e1d1.history['val_loss'])
plt.title("E1D1 Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()

plt.plot(history_e2d2.history['loss'])
plt.plot(history_e2d2.history['val_loss'])
plt.title("E2D2 Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()
# Unrepresentative Validation Dataset

plt.plot(my_history.history['loss'])
plt.plot(my_history.history['val_loss'])
plt.title("My Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()


# MAX_EPOCHS = 20
# def compile_and_fit(model):
#   early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                     patience=2,
#                                                     mode='min')
#   model.compile(loss=tf.losses.MeanSquaredError(),
#                 optimizer=tf.optimizers.Adam(),
#                 metrics=[tf.metrics.MeanAbsoluteError()])
#   history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
#                       validation_data=(X_val, y_val),
#                       callbacks=[early_stopping])
#   return history

# my_history = compile_and_fit(my_model)
# history_e1d1 = compile_and_fit(model_e1d1)
# history_e2d2 = compile_and_fit(model_e2d2)




