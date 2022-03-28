import json
import boto3
import datetime
import time
import requests

client = boto3.client('forecast')
project = 'billing_forecast_' + time.strftime('%Y%m%d%H%M', time.localtime(time.time()))


def get_or_create_iam_role(role_name):
    iam = boto3.client('iam')
    assume_role_policy_document = {
        'Version': '2012-10-17',
        'Statement': [
            {
                'Effect': 'Allow',
                'Principal': {
                    'Service': 'forecast.amazonaws.com'
                },
                'Action': 'sts:AssumeRole'
            }
        ]
    }

    try:
        create_role_response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
        )
        role_arn = create_role_response['Role']['Arn']
        print('Create Role ARN : ' + role_arn)
    except:
        print('The role ' + role_name + ' exists, ignore to create it')
        role_arn = boto3.resource('iam').Role(role_name).arn

    print('Attaching policies')
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::aws:policy/AmazonForecastFullAccess'
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess',
    )

    print('Waiting for a minute to allow IAM role policy attachment to propagate')
    time.sleep(60)

    print('Done : ' + role_arn)
    return role_arn


def get_data_ready(forecast, forecast_query, source_data_path):
    DATASET_FREQUENCY = 'D'
    TIMESTAMP_FORMAT = 'yyyy-MM-dd'
    dataset = project + '_ds'
    dataset_group = project + '_dsg'

    ## Create the dataset group
    dg_response = forecast.create_dataset_group(DatasetGroupName=dataset_group, Domain='METRICS')
    print(dg_response)
    dataset_group_arn = dg_response['DatasetGroupArn']
    forecast.describe_dataset_group(DatasetGroupArn=dataset_group_arn)

    ## Create the schema
    schema = {
        'Attributes': [
            {
                'AttributeName': 'timestamp',
                'AttributeType': 'timestamp'
            },
            {
                'AttributeName': 'metric_name',
                'AttributeType': 'string'
            },
            {
                'AttributeName': 'metric_value',
                'AttributeType': 'float'
            }
        ]
    }

    ## Create the dataset
    response = forecast.create_dataset(Domain='METRICS', DatasetType='TARGET_TIME_SERIES', DatasetName=dataset,
                                       DataFrequency=DATASET_FREQUENCY, Schema=schema)
    print(response)
    dataset_arn = response['DatasetArn']
    forecast.describe_dataset(DatasetArn=dataset_arn)

    ## Add dataset to dataset group
    forecast.update_dataset_group(DatasetGroupArn=dataset_group_arn, DatasetArns=[dataset_arn])

    ## Create IAM Role for Forecast
    # Seems unnecessary, so I'll just skip this step for now
    # It turned out to be necessary
    dataset_import_job_name = 'billing_ds_import_job_target'
    role_arn = get_or_create_iam_role('billing-forecast-role-temp')

    ## Create data import job
    ds_import_job_response = forecast.create_dataset_import_job(
        DatasetImportJobName=dataset_import_job_name,
        DatasetArn=dataset_arn,
        DataSource={
            'S3Config': {
                'Path': source_data_path,
                'RoleArn': role_arn
            }
        },
        TimestampFormat=TIMESTAMP_FORMAT
    )
    ds_import_job_arn = ds_import_job_response['DatasetImportJobArn']
    print(ds_import_job_arn)

    ## Wait till the status change from 'CREATE_IN_PROGRESS' to 'ACTIVE'
    # status_indicator = util.StatusIndicator()
    
    # while True:
    #     status = forecast.describe_dataset_import_job(DatasetImportJobArn=ds_import_job_arn)['Status']
    #     if status in ('ACTIVE', 'CREATE_FAILED'):
    #         break
    #     time.sleep(10)
    
    status = forecast.describe_dataset_import_job(DatasetImportJobArn=ds_import_job_arn)['Status']
    
    dataset_info = forecast.describe_dataset_import_job(DatasetImportJobArn=ds_import_job_arn)
    print(dataset_info)
    # payloads = notify_slack(dataset_info)
    print('================== Step 1 : DONE ==================')
    # build_predictor(forecast, forecast_query, dataset_group_arn, role_arn, s3_data_path)

    # return dataset_group_arn, role_arn, payloads
    return dataset_group_arn, role_arn, ds_import_job_arn


def notify_slack(dataset_info):
    slack_url = 'https://SLACK_URL'
    pretext = datetime.datetime.today().strftime("%y%m%d") + ' 비용 예측 현황'

    total_rows_count = str(dataset_info['FieldStatistics']['metric_value']['Count'])
    total_accounts_count = str(dataset_info['FieldStatistics']['metric_name']['CountDistinct'])
    total_null_count = str(dataset_info['FieldStatistics']['metric_value']['CountNull'])
    total_nan_count = str(dataset_info['FieldStatistics']['metric_value']['CountNan'])
    status = dataset_info['Status']  # ACTIVE / CREATE_PENDING, CREATE_FAILED

    payloads = {
        'attachments': [
            {
                'pretext': pretext,
            },
        ]
    }

    if status == 'ACTIVE':
        payloads['attachments'].append({
            'color': '#228B22',
            'fields': [
                {
                    'title': 'Data Storage',
                    'value': ':thumbsup: *Success*'
                             + '\nTotal rows : ' + total_rows_count
                             + '\nTotal accounts : ' + total_accounts_count
                             + '\nTotal null rows : ' + total_null_count + '\nTotal NaN rows : ' + total_nan_count,
                    'short': False
                }
            ]
        })

    response = requests.post(
        slack_url,
        data=json.dumps(payloads),
        headers={'Content-Type': 'application/json'}
    )

    return payloads


def lambda_handler(event, context):
    # TODO implement
    # bucket_name = event['Records'][0]['s3']['bucket']['name']
    # region_name = event['Records'][0]['awsRegion']
    # file_name = event['Records'][0]['s3']['object']['key']
    started_time = time.time()

    bucket_name = event['bucket_name']
    region_name = event['region_name']
    source_file_name = event['source_file_name']

    source_data_path = "s3://" + bucket_name + "/" + source_file_name
    session = boto3.Session(region_name=region_name)
    forecast = session.client(service_name='forecast')
    forecast_query = session.client(service_name='forecastquery')

    dataset_group_arn, role_arn, ds_import_job_arn = get_data_ready(forecast, forecast_query, source_data_path)

    pretext = time.strftime('%y%m%d', time.localtime(time.time())) + ' 비용 예측 현황'
    payloads = {
        'attachments': [
            {
                'pretext': pretext,
            },
        ]
    }
    step_info = {
        'step': 'step1',
        'started_time': started_time,
        'arn': ds_import_job_arn,
        'retry': 3,
        'count': 0
    }

    # use 'return', instead of 'callback'

    return {
        'statusCode': 200,
        'body': {
            'dataset_group_arn': dataset_group_arn,
            'role_arn': role_arn,
            'region_name': region_name,
            'project': project,
            'payloads': payloads,
            'step_info': step_info
        }
    }
