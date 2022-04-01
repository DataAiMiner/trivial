import boto3
import time

client = boto3.client('forecast')

def make_forecast(started_time, project, forecast, forecast_query, predictor_arn, role_arn):

    left_time = 800 - (time.time() - started_time)
    print('left time : ' + str(left_time))

    ## Create a forecast
    forecast_name = project + '_forecast'
    create_forecast_response = forecast.create_forecast(ForecastName=forecast_name, PredictorArn=predictor_arn)
    forecast_arn = create_forecast_response['ForecastArn']

    return forecast_arn


def lambda_handler(event, context):
    # TODO implement
    started_time = time.time()
    project = event['body']['project']
    predictor_arn = event['body']['predictor_arn']
    role_arn = event['body']['role_arn']
    region_name = event['body']['region_name']
    session = boto3.Session(region_name=region_name)
    forecast = session.client(service_name='forecast')
    forecast_query = session.client(service_name='forecastquery')

    payloads = event['body']['payloads']

    forecast_arn = make_forecast(started_time, project, forecast, forecast_query, predictor_arn, role_arn)

    step_info = {
        'step': 'step3',
        'started_time': started_time,
        'arn': forecast_arn,
        'retry': 5,
        'count': 0
    }

    return {
        'statusCode': 200,
        'body': {
            'role_arn': role_arn,
            'forecast_arn': forecast_arn,
            'region_name': region_name,
            'project': project,
            'payloads': payloads,
            'step_info': step_info
        }
    }
