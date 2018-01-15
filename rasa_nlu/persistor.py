from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
import shutil
import tarfile
import math

import boto3
import botocore
from filechunkio import FileChunkIO
from builtins import object
from rasa_nlu.config import RasaNLUConfig
from typing import Optional, Tuple, List
from typing import Text

logger = logging.getLogger(__name__)


def get_persistor(config):
    # type: (RasaNLUConfig) -> Optional[Persistor]
    """Returns an instance of the requested persistor.

    Currently, `aws` and `gcs` are supported"""

    if 'storage' not in config:
        raise KeyError("No persistent storage specified. Supported values "
                       "are {}".format(", ".join(['aws', 'gcs'])))

    if config['storage'] == 'aws':
        return AWSPersistor(config.get('aws_region'), config['bucket_name'],
                            config['aws_endpoint_url'],
                            config.get('aws_access_key_id'),
                            config.get('aws_secret_access_key')
                            )
    elif config['storage'] == 'gcs':
        return GCSPersistor(config['bucket_name'])
    elif config['storage'] == 'mutilwrite_s3':
        return MutilwriteS3Persistor(config['mutilwrite_s3_master'], config['mutilwrite_s3_slave'])
    else:
        return None


class Persistor(object):
    """Store models in cloud and fetch them when needed"""

    def persist(self, mode_directory, model_name, project, remove_tar_file=True):
        # type: (Text) -> tuple([Text, Text])
        """Uploads a model persisted in the `target_dir` to cloud storage."""

        if not os.path.isdir(mode_directory):
            raise ValueError("Target directory '{}' not "
                             "found.".format(mode_directory))

        file_key, tar_path = self._compress(mode_directory, model_name, project)
        self._persist_tar(file_key, tar_path)

        # clean up if asked
        if remove_tar_file:
            self._remove_tar_file(tar_path)

        return file_key, tar_path

    def retrieve(self, model_name, project, target_path, remove_tar_file=True):
        # type: (Text) -> None
        """Downloads a model that has been persisted to cloud storage."""

        tar_name = self._tar_name(model_name, project)

        self._retrieve_tar(tar_name)
        self._decompress(tar_name, target_path)

        # clean up if asked
        if remove_tar_file:
            self._remove_tar_file(tar_name)

    def list_models(self, project):
        # type: (Text) -> List[Text]
        """Lists all the trained models of a project."""

        raise NotImplementedError

    def list_projects(self):
        # type: () -> List[Text]
        """Lists all projects."""

        raise NotImplementedError

    def _retrieve_tar(self, filename):
        # type: (Text) -> Text
        """Downloads a model previously persisted to cloud storage."""

        raise NotImplementedError("")

    def _persist_tar(self, filekey, tarname):
        # type: (Text, Text) -> None
        """Uploads a model persisted in the `target_dir` to cloud storage."""

        raise NotImplementedError("")

    def _compress(self, model_directory, model_name, project):
        # type: (Text) -> Tuple[Text, Text]
        """Creates a compressed archive and returns key and tar."""

        base_name = self._tar_name(model_name, project, include_extension=False)
        tar_name = shutil.make_archive(base_name, 'gztar',
                                       root_dir=model_directory,
                                       base_dir=".")
        file_key = os.path.basename(tar_name)
        return file_key, tar_name

    @staticmethod
    def _project_prefix(project):
        # type: (Text) -> Text

        return '{}___'.format(project or RasaNLUConfig.DEFAULT_PROJECT_NAME)

    @staticmethod
    def _project_and_model_from_filename(filename):
        # type: (Text) -> Text

        split = filename.split("___")
        if len(split) > 1:
            model_name = split[1].replace(".tar.gz", "")
            return split[0], model_name
        else:
            return split[0], ""

    @staticmethod
    def _tar_name(model_name, project, include_extension=True):
        # type: (Text, Text, bool) -> Text

        ext = ".tar.gz" if include_extension else ""
        return '{p}{m}{ext}'.format(p=Persistor._project_prefix(project),
                                    m=model_name, ext=ext)

    @staticmethod
    def _decompress(compressed_path, target_path):
        # type: (Text, Text) -> None

        with tarfile.open(compressed_path, "r:gz") as tar:
            tar.extractall(target_path)  # project dir will be created if it not exists

    @staticmethod
    def _remove_tar_file(tar_file):
        # type: (Text) -> None
        """Remove tar file which now is useless from disk"""
        os.remove(tar_file)


def ceil_div(a, b):
    """ceil division of a / b"""
    # NOTE: all credit belongs to https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return (a + (b - 1)) // b


def chunk_yielder(element_number, max_chunk_size):
    slice_start = None
    slice_end = None
    while True:
        if slice_end == element_number:
            # is time to stop
            break

        if slice_start is None:  # for first time
            slice_start = 0
            slice_end = min(max_chunk_size, element_number)
        else:
            slice_start = slice_end
            slice_end = min(slice_end + max_chunk_size, element_number)

        yield slice_start, slice_end


def s3_file_uploader(service_client, bucket_name, file_key, file_path, chunk_size=None):
    if chunk_size is None:
        # default chunk size is 50M
        chunk_size = 50 * 1024 * 1024

    file_size = os.stat(file_path).st_size

    if file_size <= chunk_size:
        # small file, upload directly

        with open(file_path, 'rb') as fd:
            service_client.Object(bucket_name, file_key).put(Body=fd)

        # return to caller
        return None

    multipart_instance = service_client.initiate_multipart_upload(file_key)
    part_index = 1
    for start, end in chunk_yielder(file_size, chunk_size):
        bytes_length = end - start
        with FileChunkIO(file_path, 'r', offset=start, bytes=bytes_length) as fp:
            multipart_instance.upload_part_from_file(fp, part_num=part_index)
        part_index += 1

    multipart_instance.complete_upload()


class AWSPersistor(Persistor):
    """Store models on S3.

    Fetches them when needed, instead of storing them on the local disk."""

    def __init__(self, aws_region, bucket_name, endpoint_url, access_key_id=None, secret_access_key=None):
        # type: (Text, Text, Text) -> None

        super(AWSPersistor, self).__init__()
        self.s3 = boto3.resource('s3',
                                 region_name=aws_region,
                                 endpoint_url=endpoint_url,
                                 aws_access_key_id=access_key_id,
                                 aws_secret_access_key=secret_access_key
                                 )
        self._ensure_bucket_exists(bucket_name, aws_region)
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(bucket_name)

    def list_models(self, project):
        # type: (Text) -> List[Text]
        try:
            prefix = self._project_prefix(project)
            return [self._project_and_model_from_filename(obj.key)[1]
                    for obj in self.bucket.objects.filter(Prefix=prefix)]
        except Exception as e:
            logger.warn("Failed to list models for project {} in "
                        "AWS. {}".format(project, e))
            return []

    def list_projects(self):
        # type: () -> List[Text]
        try:
            projects_set = {self._project_and_model_from_filename(obj.key)[0]
                            for obj in self.bucket.objects.filter()}
            return list(projects_set)
        except Exception as e:
            logger.warning("Failed to list projects in AWS. {}".format(e))
            return []

    def _ensure_bucket_exists(self, bucket_name, aws_region):
        bucket_config = {'LocationConstraint': aws_region}
        try:
            self.s3.create_bucket(Bucket=bucket_name,
                                  CreateBucketConfiguration=bucket_config)
        except botocore.exceptions.ClientError:
            pass  # bucket already exists

    def _persist_tar(self, file_key, tar_path):
        # type: (Text, Text) -> None
        """Uploads a model persisted in the `target_dir` to s3."""

        # with open(tar_path, 'rb') as f:
        #     self.s3.Object(self.bucket_name, file_key).put(Body=f)
        s3_file_uploader(self.s3, self.bucket_name, file_key, tar_path)

    def _retrieve_tar(self, target_filename):
        # type: (Text) -> None
        """Downloads a model that has previously been persisted to s3."""

        with io.open(target_filename, 'wb') as f:
            self.bucket.download_fileobj(target_filename, f)


class GCSPersistor(Persistor):
    """Store models on Google Cloud Storage.

     Fetches them when needed, instead of storing them on the local disk."""

    def __init__(self, bucket_name):
        from google.cloud import storage

        super(GCSPersistor, self).__init__()

        self.storage_client = storage.Client()
        self._ensure_bucket_exists(bucket_name)

        self.bucket_name = bucket_name
        self.bucket = self.storage_client.bucket(bucket_name)

    def list_models(self, project):
        # type: (Text) -> List[Text]

        try:
            blob_iterator = self.bucket.list_blobs(
                    prefix=self._project_prefix(project))
            return [self._project_and_model_from_filename(b.name)[1]
                    for b in blob_iterator]
        except Exception as e:
            logger.warn("Failed to list models for project {} in "
                        "google cloud storage. {}".format(project, e))
            return []

    def list_projects(self):
        # type: () -> List[Text]

        try:
            blob_iterator = self.bucket.list_blobs()
            projects_set = {self._project_and_model_from_filename(b.name)[0]
                            for b in blob_iterator}
            return list(projects_set)
        except Exception as e:
            logger.warning("Failed to list projects in "
                           "google cloud storage. {}".format(e))
            return []

    def _ensure_bucket_exists(self, bucket_name):
        from google.cloud import exceptions

        try:
            self.storage_client.create_bucket(bucket_name)
        except exceptions.Conflict:
            # bucket exists
            pass

    def _persist_tar(self, file_key, tar_path):
        # type: (Text, Text) -> None
        """Uploads a model persisted in the `target_dir` to GCS."""

        blob = self.bucket.blob(file_key)
        blob.upload_from_filename(tar_path)

    def _retrieve_tar(self, target_filename):
        # type: (Text) -> None
        """Downloads a model that has previously been persisted to GCS."""

        blob = self.bucket.blob(target_filename)
        blob.download_to_filename(target_filename)


class MutilwriteS3Persistor(Persistor):
    """Store models on S3.

    Fetches them when needed, instead of storing them on the local disk."""

    def __init__(self, master, slaves):
        # type: (dict, List(dict)) -> None

        super(MutilwriteS3Persistor, self).__init__()

        # monkey-patched AWSPersistor: add `rollback` method
        setattr(AWSPersistor, 'rollback', self._rollback)

        self.master_node = AWSPersistor(**master)
        self.salve_nodes = [AWSPersistor(**i) for i in slaves]

    def list_models(self, project):
        # type: (Text) -> List[Text]
        return self.master_node.list_models(project)

    def list_projects(self):
        # type: (Text) -> List[Text]
        return self.master_node.list_projects()

    def persist(self, *args, **kwargs):
        persisted_nodes = []
        try:
            master_kwargs = kwargs.copy()
            master_kwargs["remove_tar_file"] = False

            logger.warning("Start to execute master persistor")

            file_key, tar_path = self.master_node.persist(*args, **master_kwargs)

            logger.warning("execute master persistor sucess")

            persisted_nodes.append(self.master_node)

            for node in self.salve_nodes:
                logger.warning("Start to execute salve persistor node: {}".format(node))

                node._persist_tar(file_key, tar_path)

                logger.warning("execute salve persistor node sucess: {}".format(node))

                persisted_nodes.append(self.master_node)

            self.master_node._remove_tar_file(tar_path)

        except Exception as e:
            # something wrong, rollback all the operation
            try:
                logger.warning("Start to rollback persistor")

                for node in persisted_nodes:

                    logger.warning("rollback {} start".format(node))

                    node.rollback(*args, **kwargs)

                    logger.warning("rollback {} successed".format(node))
                raise e
            except Exception as e:
                raise e

    @staticmethod
    def _rollback(self, mode_directory, model_name, project):
        if not os.path.isdir(mode_directory):
            raise ValueError("Target directory '{}' not found.".format(mode_directory))

        file_key, _ = self._compress(mode_directory, model_name, project, dry_run=True)

        # remove local model
        shutil.rmtree(mode_directory, ignore_errors=True)

        # remove remote model
        self.s3.Object(self.bucket_name, file_key).delete()

    @staticmethod
    def _compress(self, model_directory, model_name, project, dry_run=0):
        # type: (Text) -> Tuple[Text, Text]
        """Creates a compressed archive and returns key and tar."""

        base_name = self._tar_name(model_name, project, include_extension=False)
        tar_name = shutil.make_archive(base_name, 'gztar',
                                       root_dir=model_directory,
                                       base_dir=".",
                                       dry_run=dry_run)
        file_key = os.path.basename(tar_name)
        return file_key, tar_name
