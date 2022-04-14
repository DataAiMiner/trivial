import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from sklearn import metrics
from timeit import default_timer as timer
from pmdarima import auto_arima
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('./aiops_merge.csv')

df.reset_index(inplace=True)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')


def Augmented_Dickey_Fuller_Test_func(series, column_name):
    print(f'Results of Dickey-Fuller Test for column: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', 'No Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Conclusion:")
        print("Fail to reject the null hypothesis")
        print("series is non-stationary")


for name, column in df[[
    'disk_util-1',
    'cpu_util-1',
    'heap-1',
    'memory_util-1',
    'thread-1',
    'response_time-1',
    'disk_util-2',
    'cpu_util-2',
    'heap-2',
    'memory_util-2',
    'thread-2'
    , 'response_time-2']].iteritems():
    Augmented_Dickey_Fuller_Test_func(df[name], name)
    print('\n')

X = df[['disk_util-1', 'cpu_util-1', 'heap-1', 'memory_util-1', 'thread-1', 'response_time-1',
        'disk_util-2', 'cpu_util-2', 'heap-2', 'memory_util-2', 'thread-2', 'response_time-2']]
train, test = X[:7055], X[7055:]  # 7:3 정도로 나눔
train.head(1)

# 1차 차분 진행
train_diff = train.diff()
train_diff.dropna(inplace=True)
for name, column in train_diff[['disk_util-1', 'memory_util-1', 'disk_util-2', 'memory_util-2']].iteritems():
    Augmented_Dickey_Fuller_Test_func(train_diff[name], name)
    print('\n')

for name, column in train_diff[['disk_util-1', 'cpu_util-1', 'heap-1', 'memory_util-1', 'thread-1', 'response_time-1',
                                'disk_util-2', 'cpu_util-2', 'heap-2', 'memory_util-2', 'thread-2',
                                'response_time-2']].iteritems():
    Augmented_Dickey_Fuller_Test_func(train_diff[name], name)
    print('\n')


def cointegration_test(df):
    res = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = res.lr1
    cvts = res.cvt[:, d[str(1 - 0.05)]]

    def adjust(val, length=6):
        return str(val).ljust(length)

    print('Column Name   >  Test Stat > C(95%)    =>   Signif  \n', '--' * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), '> ', adjust(round(trace, 2), 9), ">", adjust(cvt, 8), ' =>  ', trace > cvt)


cointegration_test(train_diff[['disk_util-1', 'cpu_util-1', 'heap-1', 'memory_util-1', 'thread-1', 'response_time-1',
                               'disk_util-2', 'cpu_util-2', 'heap-2', 'memory_util-2', 'thread-2', 'response_time-2']])

pq = []
for name, column in train_diff[['disk_util-1', 'cpu_util-1', 'heap-1', 'memory_util-1', 'thread-1', 'response_time-1',
                                'disk_util-2', 'cpu_util-2', 'heap-2', 'memory_util-2', 'thread-2',
                                'response_time-2']].iteritems():
    print(f'Searching order of p and q for : {name}')
    stepwise_model = auto_arima(train_diff[name], start_p=1, start_q=1, max_p=7, max_q=7, seasonal=False,
                                trace=True, error_action='ignore', suppress_warnings=True, stepwise=True, maxiter=1000)
    parameter = stepwise_model.get_params().get('order')
    print(f'optimal order for:{name} is: {parameter} \n\n')
    pq.append(stepwise_model.get_params().get('order'))


def inverse_diff(actual_df, pred_df):
    df_res = pred_df.copy()
    columns = actual_df.columns
    for col in columns:
        df_res[str(col) + '_1st_inv_diff'] = actual_df[col].iloc[-1] + df_res[str(col)].cumsum()
    return df_res


df_results_moni = pd.DataFrame(columns=['p', 'q',
                                        'RMSE disk_util-1', 'RMSE cpu_util-1', 'RMSE heap-1', 'RMSE memory_util-1',
                                        'RMSE thread-1', 'RMSE response_time-1',
                                        'RMSE disk_util-2', 'RMSE cpu_util-2', 'RMSE heap-2', 'RMSE memory_util-2',
                                        'RMSE thread-2', 'RMSE response_time-2'
                                        ])
print('Grid Search Started')
start = timer()
for i in pq:
    if i[0] == 0 and i[2] == 0:
        pass
    else:
        print(f' Running for {i}')
        model = VARMAX(
            train_diff[['disk_util-1', 'cpu_util-1', 'heap-1', 'memory_util-1', 'thread-1', 'response_time-1',
                        'disk_util-2', 'cpu_util-2', 'heap-2', 'memory_util-2', 'thread-2', 'response_time-2']],
            order=(i[0], i[2])).fit(disp=False)
        result = model.forecast(steps=3024)
        inv_res = inverse_diff(
            df[['disk_util-1', 'cpu_util-1', 'heap-1', 'memory_util-1', 'thread-1', 'response_time-1',
                'disk_util-2', 'cpu_util-2', 'heap-2', 'memory_util-2', 'thread-2', 'response_time-2']], result)
        disk_util_1_rmse = np.sqrt(metrics.mean_squared_error(test['disk_util-1'], inv_res['disk_util-1_1st_inv_diff']))
        cpu_util_1_rmse = np.sqrt(metrics.mean_squared_error(test['cpu_util-1'], inv_res['cpu_util-1_1st_inv_diff']))
        heap_1_rmse = np.sqrt(metrics.mean_squared_error(test['heap-1'], inv_res['heap-1_1st_inv_diff']))
        memory_util_1_rmse = np.sqrt(
            metrics.mean_squared_error(test['memory_util-1'], inv_res['memory_util-1_1st_inv_diff']))
        thread_1_rmse = np.sqrt(metrics.mean_squared_error(test['thread-1'], inv_res['thread-1_1st_inv_diff']))
        response_time_1_rmse = np.sqrt(
            metrics.mean_squared_error(test['response_time-1'], inv_res['response_time-1_1st_inv_diff']))
        disk_util_2_rmse = np.sqrt(metrics.mean_squared_error(test['disk_util-2'], inv_res['disk_util-2_1st_inv_diff']))
        cpu_util_2_rmse = np.sqrt(metrics.mean_squared_error(test['cpu_util-2'], inv_res['cpu_util-2_1st_inv_diff']))
        heap_2_rmse = np.sqrt(metrics.mean_squared_error(test['heap-2'], inv_res['heap-2_1st_inv_diff']))
        memory_util_2_rmse = np.sqrt(
            metrics.mean_squared_error(test['memory_util-2'], inv_res['memory_util-2_1st_inv_diff']))
        thread_2_rmse = np.sqrt(metrics.mean_squared_error(test['thread-2'], inv_res['thread-2_1st_inv_diff']))
        response_time_2_rmse = np.sqrt(
            metrics.mean_squared_error(test['response_time-2'], inv_res['response_time-2_1st_inv_diff']))

        df_results_moni = df_results_moni.append({'p': i[0], 'q': i[2],
                                                  'RMSE disk_util-1': disk_util_1_rmse,
                                                  'RMSE cpu_util-1': cpu_util_1_rmse,
                                                  'RMSE heap-1': heap_1_rmse,
                                                  'RMSE memory_util-1': memory_util_1_rmse,
                                                  'RMSE thread-1': thread_1_rmse,
                                                  'RMSE response_time-1': response_time_1_rmse,
                                                  'RMSE disk_util-2': disk_util_2_rmse,
                                                  'RMSE cpu_util-2': cpu_util_2_rmse,
                                                  'RMSE heap-2': heap_2_rmse,
                                                  'RMSE memory_util-2': memory_util_2_rmse,
                                                  'RMSE thread-2': thread_2_rmse,
                                                  'RMSE response_time-2': response_time_2_rmse
                                                  }, ignore_index=True)
end = timer()
print(f' Total time taken to complete grid search in seconds: {(end - start)}')

print(df_results_moni.sort_values(
    by=['RMSE disk_util-1', 'RMSE cpu_util-1', 'RMSE heap-1', 'RMSE memory_util-1', 'RMSE thread-1',
        'RMSE response_time-1',
        'RMSE disk_util-2', 'RMSE cpu_util-2', 'RMSE heap-2', 'RMSE memory_util-2', 'RMSE thread-2',
        'RMSE response_time-2']))
