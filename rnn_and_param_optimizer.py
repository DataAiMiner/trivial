import numpy as np
import pandas as pd
import shutil
import os
import matplotlib.pyplot as plt

rand_number = np.array(range(1600))
np.random.shuffle(rand_number)

n_train = 1200
n_test = 400

cnt = 0
for u_id in range(1, 41):
  for s_id in range(1, 41):
    filepath_from = f'{folderpaht}/U{u_id}S{s_id}.TXT'
    if rand_number[cnt] < n_train:
      filepath_to = f'{folderpath}/train/{rand_number[cnt]}.TXT'
    else:
      filepath_to = f'{folderpath}/test/{rand_number[cnt]}.TXT'
    shutil.mode(filepath_from, filepath_to)
    cnt = cnt + 1

b_genuine = np.zeros((1600, 2))
b_genuine[:, 0] = range(1600)
cnt = 0
for u_id in range(1, 41):
  for s_id in range(1, 41):
    if s_id < 21:
      b_genuine[rand_number[cnt], 1] = 1
    else:
      b_genuine[rand_number[cnt], 1] = 0
    cnt = cnt + 1

df_genuine = pd.DataFrame(b_genuine)
df_genuine.columnns = ['ID', 'bGenuine']
df_genuine = df_genuine.astype('int32')
df_genuine.to_csv(f'{folderpath}.train_info.csv', index=False)


def get_max_len(folderpath_train, folderpath_test):
  n_max_len = 0
  for filename in os.listdir(folderpath_train):
    filepath = f'{folderpath_train}/{filename}'
    d, tmp = load_a_sig(filepath)
    in d.shape[0] > n_max_len:
      n_max_len = d.shape[0]
  
  for filename in os.listdir(folderpath_test):
    filepath = f'{folderpath_test}/{filename}'
    d, tmp = load_a_sig(filepath)
    if d.shape[0] > n_max_len:
      n_max_len = d.shape[0]
  
  return n_max_len


def load_data(folderpath, n_max_len, sig_ids):
  filelist = os.listdir(folderpath)
  n_files = len(filelist)
  d = np.zeros((n_files, n_max_len, 2))
  len_sign = np.zeros(n_files)
  
  for i in sig_ids:
    filepath = f'{folderpath}/{i}.TXT'
    d_tmp, len_sign[i-sig_ids[0]] = load_a_sig(filepath)
    d[i-sig_ids[0], :d_tmp.shape[0], :] = d_tmp
  
  return d, len_sign


def load_a_sig(filepath):
  f = open(filepath, 'rt')
  nPoints = int(f.readline())
  d = np.zeros((nPoints, 2))
  for i in range(nPoints):
    line = f.readline()
    toks = line.split(' ')
    d[i, :] = [int(toks[0]), int(toks[1])]
  f.close()
  return d, nPoints


n_max_len = get_max_len(f'{folerpath}/train', f'{folderpath}/test')
x_train, x_traiin_len = load_data(f'{folderpath}/train', n_max_len, range(n_train))
x_test, x_test_len = load_data(f'{folderpath}/test', n_max_len, range(n_train, 1600))

s_id = 10
len_sig = int(x_train_len[s_id])
plt.plot(x_train[s_id, :len_sig, 0], x_train[s_id, :len_sig, 1])

for i in range(x_train.shape[0]):
  len_tmp = int(x_train_len[i])
  h_min = np.min(x_train[i, :len_tmp, 0])
  h_max = np.max(x_train[i, :len_tmp, 0])
  v_min = np.min(x_train[i, :len_tmp, 1])
  v_max = np.max(x_train[i, :len_tmp, 1])
  w_h_ratio = (v_max-v_min)/(h_max-h_min)
  x_train[i, :len_tmp, 0] = (x_train[i, :len_tmp, 0] - h_min) / (h_max-h_min)
  x_train[i, :len_tmp, 1] = (x_train[i, :len_tmp, 1] - v_min) / (v_max-v_min) + w_h_ratio

for i in range(x_test.shape[0]):
  len_tmp = int(x_test_len[i])
  h_min = np.min(x_test[i, :len_tmp, 0])
  h_max = np.max(x_test[i, :len_tmp, 0])
  v_min = np.min(x_test[i, :len_tmp, 1])
  v_max = np.max(x_test[i, :len_tmp, 1])
  w_h_ratio = (v_max-v_min)/(h_max-h_min)
  x_test[i, :len_tmp, 0] = (x_test[i, :len_tmp, 0] - h_min) / (h_max-h_min)
  x_test[i, :len_tmp, 1] = (x_test[i, :len_tmp, 1] - v_min) / (v_max-v_min) + w_h_ratio

s_id = 10
plt.plot(x_train[s_id, :, 0], x_train[s_id, :, 1])
