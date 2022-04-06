import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('aiops_merge.csv')
df_noheap = pd.read_csv('aiops_merge_noheap.csv')

df.reset_index(inplace=True)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

df_noheap.reset_index(inplace=True)
df_noheap['time'] = pd.to_datetime(df_noheap['time'])
df_noheap = df_noheap.set_index('time')

df.head(2)
df_noheap.head(2)

df.plot(figsize=(30,20))
df_noheap.plot(figsize=(30,20))

df.dtypes

df_lena01 = pd.DataFrame(df[['cpu_util-1', 'disk_util-1','heap-1','thread-1','memory_util-1','response_time-1']], index=df.index)
df_lena01.dtypes
df_lena01.head()
df_lena02 = pd.DataFrame(df[['cpu_util-2', 'disk_util-2','heap-2','thread-2','memory_util-2','response_time-2']], index=df.index)
df_lena02.dtypes
df_lena02.head()

# 빈도 설정하기
# https://stackoverflow.com/questions/35339139/what-values-are-valid-in-pandas-freq-tags
# https://dsbook.tistory.com/267
df_lena01.asfreq(freq='Min')[df_lena01.asfreq('Min').isnull().sum(axis=1) > 0]
df_lena01.asfreq(freq='Min').isnull().sum()
df_lena01.index
df_lena01 = df_lena01.asfreq('Min')
df_lena01.index

# In seasonal_decompose we have to set the model. We can either set the model to be Additive or Multiplicative. 
# A rule of thumb for selecting the right model is to see in our plot if the trend and seasonal variation are relatively constant over time, in other words, linear.
# If yes, then we will select the Additive model. 
# Otherwise, if the trend and seasonal variation increase or decrease over time then we use the Multiplicative model.
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html?highlight=seasonal_decompose
decomp_freq = 24*60
result_rt_lena01 = seasonal_decompose(x=df_lena01['response_time-1'], model='additive')  # ValueError: freq T not understood. Please report if you think this is in error.
result_rt_lena01.plot()
