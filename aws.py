"""
AWS API with boto3
"""
import boto3
import os
import tempfile
from PIL import Image


class Aws():
    def __init__(self, boto3_session):
        super(Aws, self).__init__()

        self.sqs = boto3_session.resource(service_name='sqs', api_version='2012-11-05', region_name='ap-northeast-2')
        self.request_queue = self.sqs.get_queue_by_name(QueueName=os.environ['A_QUEUE'])

        self.s3 = boto3_session.resource(service_name='s3', api_version='2006-03-01', region_name='ap-northeast-2')
        self.a_bucket = self.s3.Bucket(os.environ['A_BUCKET'])


BOTO3_SESSION = boto3.Session(profile_name=os.environ['AWS_PROFILE'])
print('boto3 profile name: ', BOTO3_SESSION.profile_name)
AWS = Aws(boto3_session=BOTO3_SESSION)


def fetch_request(timeout_seconds: int=20):
    return AWS.request_queue.receive_messages(AttributeNames=['All'], WaitTimeSeconds=timeout_seconds)


def load_input(key: str, key_prefix: str='inputs/'):
    temp_fd, temp_path = tempfile.mkstemp()
    # temp_path = './input'
    try:
        AWS.a_bucket.download_file('{}{}'.format(key_prefix, key), temp_path)
        # return Image.open(open(temp_path))
        return Image.open(temp_path).convert('RGB')
    except Exception:
        raise
    finally:
        pass
        os.close(temp_fd)
        os.unlink(temp_path)


def save_output(key: str, file_path: str, key_prefix: str='outputs/'):
    try:
        AWS.a_bucket.upload_file(file_path, '{}{}'.format(key_prefix, key))
    except Exception:
        raise
    finally:
        pass