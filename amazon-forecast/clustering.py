import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
from copy import deepcopy

THRESHOLD = 1.0


def find_dates(longest_acntId, grouped_data):
    for key, row in grouped_data:
        if key == longest_acntId:
            return row['date']


def find_longest(dataset):
    length_record = 0
    longest_acnt = ''
    max_len = 0
    for acntId, row in dataset:
        if len(row) > length_record:
            length_record = len(row)
            longest_acnt = acntId
            max_len = len(row)
    return longest_acnt, max_len


def fill_in(new_dataset, dates, id, row):
    series = pd.Series(row['cost'].tolist(), index=row['date'].tolist())
    new_series = series.reindex(dates.tolist())
    return pd.DataFrame(new_series, columns=[id])


def main():
    max_len = 0  # length of the account with longest data
    dataset = pd.read_csv('billing_data_1130_v2.csv', header=None
                          , dtype={'date': 'str', 'acntId': 'str', 'cost': 'float'})
    dataset.columns = ['date', 'acntId', 'cost']

    '''
    NORMALIZATION
    '''
    data = dataset.groupby('acntId').agg(list)['cost']
    acnt_ids = np.unique(dataset.to_numpy()[:, 1])
    grouped_dataset = dataset.groupby(dataset['acntId'])
    longest_acntId, max_len = find_longest(grouped_dataset)
    dates = find_dates(longest_acntId, grouped_dataset).values
    
    new_dataset = pd.DataFrame(index=dates, columns=acnt_ids)
    new_data = pd.DataFrame(index=dates)  # aligns dates

    for id, row in grouped_dataset:
        new_row = fill_in(new_dataset, dates, id, row)
        new_data = pd.concat([new_data, new_row], axis=1)
    transposed_filled_data = pd.DataFrame.transpose(new_data.copy())

    scaler = MinMaxScaler()
    scaled_dataset = transposed_filled_data.copy()
    scaled_dataset[:] = scaler.fit_transform(scaled_dataset[:])

    '''
    FORMING A CLUSTERABLE DATASET
    '''
    trajectoriesSet = {}
    for ind in scaled_dataset.index:
        adding_list = []
        for col_ind in scaled_dataset.columns:
            val = scaled_dataset.loc[ind, col_ind]
            if not np.isnan(val):
                adding_list.append(val)
        trajectoriesSet[(str(ind),)] = [np.array(adding_list)]


    '''
    CLUSTERING !! Error occurred, need improving !!
    '''
    trajectories = deepcopy(trajectoriesSet)
    distanceMatrixDictionary = {}
    iteration = 1
    while True:
        distanceMatrix = np.empty((len(trajectories), len(trajectories),))
        distanceMatrix[:] = np.nan

        for index1, (filter1, trajectory1) in enumerate(trajectories.items()):

            for index2, (filter2, trajectory2) in enumerate(trajectories.items()):

                if index1 > index2:
                    continue

                elif index1 == index2:
                    continue

                else:
                    unionFilter = filter1 + filter2
                    sorted(unionFilter)

                    if unionFilter in distanceMatrixDictionary.keys():
                        distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)
                        continue

                    metric = []
                    for subItem1 in trajectory1:
                        for subItem2 in trajectory2:
                            metric.append(dtw.distance(subItem1, subItem2, psi=1))

                    metric = max(metric)

                    distanceMatrix[index1][index2] = metric
                    distanceMatrixDictionary[unionFilter] = metric

        minValue = np.min(list(distanceMatrixDictionary.values()))

        if minValue > THRESHOLD:
            break

        minIndices = np.where(distanceMatrix == minValue)
        minIndices = list(zip(minIndices[0], minIndices[1]))

        minIndex = minIndices[0]

        filter1 = list(trajectories.keys())[minIndex[0]]
        filter2 = list(trajectories.keys())[minIndex[1]]

        trajectory1 = trajectories.get(filter1)
        trajectory2 = trajectories.get(filter2)

        unionFilter = filter1 + filter2
        sorted(unionFilter)

        trajectoryGroup = trajectory1 + trajectory2

        trajectories = {key: value for key, value in trajectories.items()
                        if all(value not in unionFilter for value in key)}

        distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                    if all(value not in unionFilter for value in key)}

        trajectories[unionFilter] = trajectoryGroup

        print(iteration, 'finished!')
        iteration += 1

        if len(list(trajectories.keys())) == 1:
            break

    for key, _ in trajectories.items():
        print(key)
