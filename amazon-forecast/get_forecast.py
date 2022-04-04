import boto3
import time
import datetime

client = boto3.client('forecast')


def get_forecast(project, forecast, forecast_arn, role_arn):
    saving_data_path = 's3://cx-prod-cost-prediction/prediction-result/billing_forecast_result_' + time.strftime('%y%m%d', time.localtime(time.time()))
    forecast_export_job_name = project + '_export'
    forecast_export_job_response = forecast.create_forecast_export_job(
        ForecastExportJobName=forecast_export_job_name,
        ForecastArn=forecast_arn,
        Destination={
            'S3Config': {
                'Path': saving_data_path,
                'RoleArn': role_arn,
            }
        }
    )
    forecast_export_job_arn = forecast_export_job_response['ForecastExportJobArn']

    return forecast_export_job_arn


def lambda_handler(event, context):
    # TODO implement
    started_time = time.time()
    project = event['body']['project']
    forecast_arn = event['body']['forecast_arn']
    role_arn = event['body']['role_arn']
    region_name = event['body']['region_name']
    session = boto3.Session(region_name=region_name)
    forecast = session.client(service_name='forecast')
    forecast_query = session.client(service_name='forecastquery')

    payloads = event['body']['payloads']

    forecast_export_job_arn = get_forecast(project, forecast, forecast_arn, role_arn)

    step_info = {
        'step': 'step4',
        'started_time': started_time,
        'arn': forecast_export_job_arn,
        'retry': 1,
        'count': 0
    }

    return {
        'statusCode': 200,
        'body': {
            'region_name': region_name,
            'payloads': payloads,
            'step_info': step_info
        }
    }
