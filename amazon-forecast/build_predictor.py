import boto3
import time
import json

client = boto3.client('forecast')


def build_predictor(started_time, project, forecast, dataset_group_arn, role_arn):
    ## Create a predictor
    left_time = 800 - (time.time() - started_time)
    print('started time : ' + str(started_time))
    print('current time : ' + str(time.time()))
    print('left time : ' + str(left_time))
    predictor_name = project + '_NPTS_predictor'
    forecast_horizon = 60
    algorithm_arn = 'arn:aws:forecast:::algorithm/NPTS'

    create_predictor_response = forecast.create_predictor(
        PredictorName=predictor_name,
        AlgorithmArn=algorithm_arn,
        ForecastHorizon=forecast_horizon,
        PerformAutoML=False,
        PerformHPO=False,
        EvaluationParameters={
            'NumberOfBacktestWindows': 1,
            'BackTestWindowOffset': 60
        },
        InputDataConfig={
            'DatasetGroupArn': dataset_group_arn
        },
        FeaturizationConfig={
            'ForecastFrequency': 'D',
            'Featurizations': [
                {
                    'AttributeName': 'metric_value',
                    'FeaturizationPipeline': [
                        {
                            'FeaturizationMethodName': 'filling',
                            'FeaturizationMethodParameters': {
                                'frontfill': 'none',
                                'middlefill': 'nan',
                                'backfill': 'nan'
                            }
                        }
                    ]
                }
            ]
        }
    )
    predictor_arn = create_predictor_response['PredictorArn']
    print(predictor_arn)

    status = forecast.describe_predictor(PredictorArn=predictor_arn)['Status']
    print(status)
    return predictor_arn


def lambda_handler(event, context):
    # TODO implement
    started_time = time.time()
    project = event['body']['project']
    dataset_group_arn = event['body']['dataset_group_arn']
    role_arn = event['body']['role_arn']
    region_name = event['body']['region_name']
    session = boto3.Session(region_name=region_name)
    forecast = session.client(service_name='forecast')

    payloads = event['body']['payloads']

    predictor_arn = build_predictor(started_time, project, forecast, dataset_group_arn, role_arn)
    # evaluate_predictor(forecast, forecast_response, forecast_arn, role_arn, s3_data_path)
    step_info = {
        'step': 'step2',
        'started_time': started_time,
        'arn': predictor_arn,
        'retry': 6,
        'count': 0
    }

    return {
        'statusCode': 200,
        'body': {
            'role_arn': role_arn,
            'predictor_arn': predictor_arn,
            'region_name': region_name,
            'project': project,
            'payloads': payloads,
            'step_info': step_info
        }
    }
