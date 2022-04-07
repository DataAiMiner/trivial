import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# df = pd.read_csv('aiops_merge.csv', index_col=0)
# df_noheap = pd.read_csv('aiops_merge_noheap.csv', index_col=0)
df = pd.read_csv('./drive/MyDrive/Colab Notebooks/aiops_merge.csv')
df_noheap = pd.read_csv('./drive/MyDrive/Colab Notebooks/aiops_merge_noheap.csv')

df.reset_index(inplace=True)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

df_noheap.reset_index(inplace=True)
df_noheap['time'] = pd.to_datetime(df_noheap['time'])
df_noheap = df_noheap.set_index('time')

df_lena01 = pd.DataFrame(df[['cpu_util-1', 'disk_util-1','heap-1','thread-1','memory_util-1','response_time-1']], index=df.index)
df_lena01.dtypes
df_lena01.head()

df_lena02 = pd.DataFrame(df[['cpu_util-2', 'disk_util-2','heap-2','thread-2','memory_util-2','response_time-2']], index=df.index)
df_lena02.dtypes
df_lena02.head()

df_lena01 = df_lena01.asfreq('Min')
df_lena01.index

# In seasonal_decompose we have to set the model. We can either set the model to be Additive or Multiplicative. 
# A rule of thumb for selecting the right model is to see in our plot if the trend and seasonal variation are relatively constant over time, in other words, linear.
# If yes, then we will select the Additive model. 
# Otherwise, if the trend and seasonal variation increase or decrease over time then we use the Multiplicative model.
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html?highlight=seasonal_decompose
decomp_freq = 24*60
result_rt_lena01 = seasonal_decompose(x=df_lena01['response_time-1'], model='additive', period=decomp_freq)
result_rt_lena01.plot()
result_rt_lena01_cpu = seasonal_decompose(x=df_lena01['cpu_util-1'], model='additive', period=decomp_freq)
result_rt_lena01_cpu.plot()
result_rt_lena01_disk = seasonal_decompose(x=df_lena01['disk_util-1'], model='additive', period=decomp_freq)
result_rt_lena01_disk.plot()
result_rt_lena01_thread = seasonal_decompose(x=df_lena01['thread-1'], model='additive', period=decomp_freq)
result_rt_lena01_thread.plot()
result_rt_lena01_memory = seasonal_decompose(x=df_lena01['memory_util-1'], model='additive', period=decomp_freq)
result_rt_lena01_memory.plot()
result_rt_lena01_heap = seasonal_decompose(x=df_lena01['heap-1'], model='additive', period=decomp_freq)
result_rt_lena01_heap.plot()

# Multiplicative seasonality is not appropriate for zero and negative values
# result_rt_lena01 = seasonal_decompose(x=df_lena01['response_time-1'], model='multiplicative', period=decomp_freq)
# result_rt_lena01.plot()
result_rt_lena01_cpu_m = seasonal_decompose(x=df_lena01['cpu_util-1'], model='multiplicative', period=decomp_freq)
result_rt_lena01_cpu_m.plot()
result_rt_lena01_disk_m = seasonal_decompose(x=df_lena01['disk_util-1'], model='multiplicative', period=decomp_freq)
result_rt_lena01_disk_m.plot()
# result_rt_lena01_thread = seasonal_decompose(x=df_lena01['thread-1'], model='multiplicative', period=decomp_freq)
# result_rt_lena01_thread.plot()
result_rt_lena01_memory_m = seasonal_decompose(x=df_lena01['memory_util-1'], model='multiplicative', period=decomp_freq)
result_rt_lena01_memory_m.plot()
# result_rt_lena01_heap = seasonal_decompose(x=df_lena01['heap-1'], model='multiplicative', period=decomp_freq)
# result_rt_lena01_heap.plot()

result_rt_lena02_memory = seasonal_decompose(x=df_lena02['response_time-2'], model='additive', period=decomp_freq)
result_rt_lena02_memory.plot()
result_rt_lena02_cpu = seasonal_decompose(x=df_lena02['cpu_util-2'], model='additive', period=decomp_freq)
result_rt_lena02_cpu.plot()
result_rt_lena02_disk = seasonal_decompose(x=df_lena02['disk_util-2'], model='additive', period=decomp_freq)
result_rt_lena02_disk.plot()
result_rt_lena02_thread = seasonal_decompose(x=df_lena02['thread-2'], model='additive', period=decomp_freq)
result_rt_lena02_thread.plot()
result_rt_lena02_memory = seasonal_decompose(x=df_lena02['memory_util-2'], model='additive', period=decomp_freq)
result_rt_lena02_memory.plot()
result_rt_lena02_heap = seasonal_decompose(x=df_lena02['heap-2'], model='additive', period=decomp_freq)
result_rt_lena02_heap.plot()

# Multiplicative seasonality is not appropriate for zero and negative values
# result_rt_lena02_memory = seasonal_decompose(x=df_lena02['response_time-2'], model='multiplicative', period=decomp_freq)
# result_rt_lena02_memory.plot()
result_rt_lena02_cpu_m = seasonal_decompose(x=df_lena02['cpu_util-2'], model='multiplicative', period=decomp_freq)
result_rt_lena02_cpu_m.plot()
result_rt_lena02_disk_m = seasonal_decompose(x=df_lena02['disk_util-2'], model='multiplicative', period=decomp_freq)
result_rt_lena02_disk_m.plot()
# result_rt_lena02_thread = seasonal_decompose(x=df_lena02['thread-2'], model='multiplicative', period=decomp_freq)
# result_rt_lena02_thread.plot()
result_rt_lena02_memory_m = seasonal_decompose(x=df_lena02['memory_util-2'], model='multiplicative', period=decomp_freq)
result_rt_lena02_memory_m.plot()
# result_rt_lena02_heap = seasonal_decompose(x=df_lena02['heap-2'], model='multiplicative', period=decomp_freq)
# result_rt_lena02_heap.plot()

from statsmodels.tsa.stattools import adfuller

# p-value(0과 1 사이의 값)가 유의수준(보통 0.05)보다 작으면 단위근이 존재하지 않는다(=시계열이 정상성을 가진다)고 본다.
# H0: 시계열이 단위근을 가진다.
def adf_test(df):
  result = adfuller(df.values)
  print('ADF statistics : %f' % result[0])
  print('p-value: %f' % result[0])
  print('Critical values: ')
  for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

    
print('ADF Test: disk_util-1')
adf_test(df_lena01['disk_util-1'])
print('===================================')
print('ADF Test: cpu_util-1')
adf_test(df_lena01['cpu_util-1'])
print('===================================')
print('ADF Test: heap-1')
adf_test(df_lena01['heap-1'])
print('===================================')
print('ADF Test: memory_util-1')
adf_test(df_lena01['memory_util-1'])
print('===================================')
print('ADF Test: thread-1')
adf_test(df_lena01['thread-1'])
print('===================================')
print('ADF Test: response_time-1')
print('ADF Test: disk_util-2')
adf_test(df_lena02['disk_util-2'])
print('===================================')
print('ADF Test: cpu_util-2')
adf_test(df_lena02['cpu_util-2'])
print('===================================')
print('ADF Test: heap-2')
adf_test(df_lena02['heap-2'])
print('===================================')
print('ADF Test: memory_util-2')
adf_test(df_lena02['memory_util-2'])
print('===================================')
print('ADF Test: thread-2')
adf_test(df_lena02['thread-2'])
print('===================================')
print('ADF Test: response_time-2')
adf_test(df_lena02['response_time-2'])
adf_test(df_lena01['response_time-1'])
