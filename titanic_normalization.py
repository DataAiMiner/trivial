import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
#!kaggle competition download -c titanic


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_train_np = np.zeros([data_test.shape[0], 8])
cnt = 0
sex_num = np.zeros(data_train_np.shape[0])
sex_num[np.array(data_train['Sex'] == 'female')] = 1
# for d in data_train['Sex']:
#     if d == 'female':
#         sex_num[cnt] = 1
#         cnt += 1

data_train_np[:, 0:2] = to_categorical(sex_num)
data_train_np[:, 2:5] = to_categorical(data_train['Pclass'].to_numpy()-1)
data_train_np[:, 5] = data_train['Age']/80
data_train_np[:, 6] = data_train['SibSp']/10
data_train_np[:, 7] = data_train['Parch']/10

data_train_np[np.isnan(data_train_np)] = 30/80

data_train_np_y = to_categorical(data_train['Survived'])
