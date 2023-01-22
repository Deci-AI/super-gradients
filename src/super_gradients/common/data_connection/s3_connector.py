import os
import sys
from io import StringIO, BytesIO
from typing import List

import botocore

from super_gradients.common.aws_connection.aws_connector import AWSConnector
from super_gradients.common.decorators.explicit_params_validator import explicit_params_validation
from super_gradients.common.abstractions.abstract_logger import ILogger


class KeyNotExistInBucketError(Exception):
    pass


class S3Connector(ILogger):
    """
    S3Connector - S3 Connection Manager
    """

    def __init__(self, env: str, bucket_name: str):
        """
        :param s3_bucket:
        """
        super().__init__()
        self.env = env
        self.bucket_name = bucket_name
        self.s3_client = AWSConnector.get_aws_client_for_service_name(profile_name=env, service_name="s3")
        self.s3_resource = AWSConnector.get_aws_resource_for_service_name(profile_name=env, service_name="s3")

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def check_key_exists(self, s3_key_to_check: str) -> bool:
        """
        check_key_exists - Checks if an S3 key exists
        :param s3_key_to_check:
        :return:
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key_to_check)
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] == "404":
                return False
            else:
                self._logger.error("Failed to check key: " + str(s3_key_to_check) + " existence in bucket" + str(self.bucket_name))
                return None
        else:
            return True

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def get_object_by_etag(self, bucket_relative_file_name: str, etag: str) -> object:
        """
        get_object_by_etag - Gets S3 object by it's ETag heder if it. exists
        :param bucket_relative_file_name: The name of the file in the bucket (relative)
        :param etag: The ETag of the object in S3
        :return:
        """
        try:
            etag = etag.strip('"')
            s3_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=bucket_relative_file_name, IfMatch=etag)
            return s3_object
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] == "404":
                return False
            else:
                self._logger.error("Failed to check ETag: " + str(etag) + " existence in bucket " + str(self.bucket_name))
        return

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def create_bucket(self) -> bool:
        """
        Creates a bucket with the initialized bucket name.
        :return: The new bucket response
        :raises ClientError: If the creation failed for any reason.
        """
        try:
            # TODO: Change bucket_owner_arn to the company's proper IAM Role
            self._logger.info("Creating Bucket: " + self.bucket_name)
            create_bucket_response = self.s3_client.create_bucket(ACL="private", Bucket=self.bucket_name)
            self._logger.info(f"Successfully created bucket: {create_bucket_response}")

            # Changing the bucket public access block to be private (disable public access)
            self._logger.debug("Disabling public access to the bucket...")
            self.s3_client.put_public_access_block(
                PublicAccessBlockConfiguration={"BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True},
                Bucket=self.bucket_name,
            )
            return create_bucket_response
        except botocore.exceptions.ClientError as err:
            self._logger.fatal(f'Failed to create bucket "{self.bucket_name}": {err}')
            raise

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def delete_bucket(self):
        """
        Deletes a bucket with the initialized bucket name.
        :return: True if succeeded.
        :raises ClientError: If the creation failed for any reason.
        """
        try:
            self._logger.info("Deleting Bucket: " + self.bucket_name + " from S3")
            bucket = self.s3_resource.Bucket(self.bucket_name)
            bucket.objects.all().delete()
            bucket.delete()
            self._logger.debug("Successfully Deleted Bucket: " + self.bucket_name + " from S3")
        except botocore.exceptions.ClientError as ex:
            self._logger.fatal(f"Failed to delete bucket {self.bucket_name}: {ex}")
            raise ex
        return True

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def get_object_metadata(self, s3_key: str):
        try:
            return self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] == "404":
                msg = "[" + sys._getframe().f_code.co_name + "] - Key does not exist in bucket)"
                self._logger.error(msg)
                raise KeyNotExistInBucketError(msg)
            raise ex

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def delete_key(self, s3_key_to_delete: str) -> bool:
        """
        delete_key - Deletes a Key from an S3 Bucket
            :param s3_key_to_delete:
            :return: True/False if the operation succeeded/failed
        """
        try:
            self._logger.debug("Deleting Key: " + s3_key_to_delete + " from S3 bucket: " + self.bucket_name)
            obj_status = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key_to_delete)
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] == "404":
                self._logger.error("[" + sys._getframe().f_code.co_name + "] - Key does not exist in bucket)")
            return False

        if obj_status["ContentLength"]:
            self._logger.debug("[" + sys._getframe().f_code.co_name + "] - Deleting file s3://" + self.bucket_name + s3_key_to_delete)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key_to_delete)

        return True

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def upload_file_from_stream(self, file, key: str):
        """
        upload_file - Uploads a file to S3 via boto3 interface
                      *Please Notice* - This method is for working with files, not objects
            :param key: The key (filename) to create in the S3 bucket
            :param filen: File to upload
            :return True/False if the operation succeeded/failed
        """
        try:
            self._logger.debug("Uploading Key: " + key + " to S3 bucket: " + self.bucket_name)
            buffer = BytesIO(file)
            self.upload_buffer(key, buffer)
            return True
        except Exception as ex:
            self._logger.critical("[" + sys._getframe().f_code.co_name + "] - Caught Exception while trying to upload file " + str(key) + "to S3" + str(ex))
            return False

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def upload_file(self, filename_to_upload: str, key: str):
        """
        upload_file - Uploads a file to S3 via boto3 interface
                      *Please Notice* - This method is for working with files, not objects
            :param key:                The key (filename) to create in the S3 bucket
            :param filename_to_upload: Filename of the file to upload
            :return True/False if the operation succeeded/failed
        """
        try:
            self._logger.debug("Uploading Key: " + key + " to S3 bucket: " + self.bucket_name)

            self.s3_client.upload_file(Bucket=self.bucket_name, Filename=filename_to_upload, Key=key)
            return True

        except Exception as ex:
            self._logger.critical(f"[{sys._getframe().f_code.co_name}] - Caught Exception while trying to upload file {filename_to_upload} to S3 {ex}")
            return False

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def download_key(self, target_path: str, key_to_download: str) -> bool:
        """
        download_file - Downloads a key from S3 using boto3 to the provided filename
                        Please Notice* - This method is for working with files, not objects
            :param key_to_download:    The key (filename) to download from the S3 bucket
            :param target_path:           Filename of the file to download the content of the key to
            :return:                   True/False if the operation succeeded/failed
        """
        try:
            self._logger.debug("Uploading Key: " + key_to_download + " from S3 bucket: " + self.bucket_name)
            self.s3_client.download_file(Bucket=self.bucket_name, Filename=target_path, Key=key_to_download)
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] == "404":
                self._logger.error("[" + sys._getframe().f_code.co_name + "] - Key does exist in bucket)")
            else:
                self._logger.critical(f"[{sys._getframe().f_code.co_name}] - Caught Exception while trying to download key {key_to_download} from S3 {ex}")
            return False

        return True

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def download_keys_by_prefix(self, s3_bucket_path_prefix: str, local_download_dir: str, s3_file_path_prefix: str = ""):
        """
        download_keys_by_prefix - Download all of the keys who match the provided in-bucket path prefix and file prefix
            :param s3_bucket_path_prefix:   The S3 "folder" to download from
            :param local_download_dir:      The local directory to download the files to
            :param s3_file_path_prefix:     The specific prefix of the files we want to download
        :return:
        """
        if not os.path.isdir(local_download_dir):
            raise ValueError("[" + sys._getframe().f_code.co_name + "] - Provided directory does not exist")

        paginator = self.s3_client.get_paginator("list_objects")
        prefix = s3_bucket_path_prefix if not s3_file_path_prefix else s3_bucket_path_prefix + "/" + s3_file_path_prefix
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        for item in page_iterator.search("Contents"):
            if item is not None:
                if item["Key"] == s3_bucket_path_prefix:
                    continue
            key_to_download = item["Key"]
            local_filename = key_to_download.split("/")[-1]
            self.download_key(target_path=local_download_dir + "/" + local_filename, key_to_download=key_to_download)

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def download_file_by_path(self, s3_file_path: str, local_download_dir: str):
        """
        :param s3_file_path: str - path ot s3 file e.g./ "s3://x/y.zip"
        :param local_download_dir: path to download
        :return:
        """

        if not os.path.isdir(local_download_dir):
            raise ValueError("[" + sys._getframe().f_code.co_name + "] - Provided directory does not exist")

        local_filename = s3_file_path.split("/")[-1]
        self.download_key(target_path=local_download_dir + "/" + local_filename, key_to_download=s3_file_path)
        return local_filename

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def empty_folder_content_by_path_prefix(self, s3_bucket_path_prefix) -> list:
        """
        empty_folder_content_by_path_prefix - Deletes all of the files in the specified bucket path
            :param s3_bucket_path_prefix: The "folder" to empty
            :returns: Errors list
        """
        paginator = self.s3_client.get_paginator("list_objects")
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_bucket_path_prefix)

        files_dict_to_delete = dict(Objects=[])
        errors_list = []

        for item in page_iterator.search("Contents"):
            if item is not None:
                if item["Key"] == s3_bucket_path_prefix:
                    continue
                files_dict_to_delete["Objects"].append(dict(Key=item["Key"]))

                # IF OBJECTS LIMIT HAS BEEN REACHED, FLUSH
                if len(files_dict_to_delete["Objects"]) >= 1000:
                    self._delete_files_left_in_list(errors_list, files_dict_to_delete)
                    files_dict_to_delete = dict(Objects=[])

        # DELETE THE FILES LEFT IN THE LIST
        if len(files_dict_to_delete["Objects"]):
            self._delete_files_left_in_list(errors_list, files_dict_to_delete)

        return errors_list

    def _delete_files_left_in_list(self, errors_list, files_dict_to_delete):
        try:
            s3_response = self.s3_client.delete_objects(Bucket=self.bucket_name, Delete=files_dict_to_delete)
        except Exception as ex:
            self._logger.critical("[" + sys._getframe().f_code.co_name + "] - Caught Exception while trying to delete keys " + "from S3 " + str(ex))
        if "Errors" in s3_response:
            errors_list.append(s3_response["Errors"])

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def upload_buffer(self, new_key_name: str, buffer_to_write: StringIO):
        """
        Uploads a buffer into a file in S3 with the provided key name.
        :bucket: The name of the bucket
        :new_key_name: The name of the file to create in s3
        :buffer_to_write: A buffer that contains the file contents.
        """
        self.s3_resource.Object(self.bucket_name, new_key_name).put(Body=buffer_to_write.getvalue())

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def list_bucket_objects(self, prefix: str = None) -> List[dict]:
        """
        Gets a list of dictionaries, representing files in the S3 bucket that is passed in the constructor (self.bucket).
        :param prefix: A prefix filter for the files names.
        :return: the objects, dict as received from botocore.
        """
        paginator = self.s3_client.get_paginator("list_objects")
        if prefix:
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        else:
            page_iterator = paginator.paginate(Bucket=self.bucket_name)

        bucket_objects = []
        for item in page_iterator.search("Contents"):
            if not item or item["Key"] == self.bucket_name:
                continue
            bucket_objects.append(item)
        return bucket_objects

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def create_presigned_upload_url(self, object_name: str, fields=None, conditions=None, expiration=3600):
        """Generate a presigned URL S3 POST request to upload a file
        :param bucket_name: string
        :param object_name: string
        :param fields: Dictionary of prefilled form fields
        :param conditions: List of conditions to include in the policy
        :param expiration: Time in seconds for the presigned URL to remain valid
        :return: Dictionary with the following keys:
            url: URL to post to
            fields: Dictionary of form fields and values to submit with the POST request
        """
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html#generating-a-presigned-url-to-upload-a-file
        file_already_exist = self.check_key_exists(object_name)
        if file_already_exist:
            raise FileExistsError(f"The key {object_name} already exists in bucket {self.bucket_name}")

        response = self.s3_client.generate_presigned_post(self.bucket_name, object_name, Fields=fields, Conditions=conditions, ExpiresIn=expiration)
        return response

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def create_presigned_download_url(self, bucket_name: str, object_name: str, expiration=3600):
        """Generate a presigned URL S3 Get request to download a file
        :param bucket_name: string
        :param object_name: string
        :param expiration: Time in seconds for the presigned URL to remain valid
        :return: URL encoded with the credentials in the query, to be fetched using any HTTP client.
        """
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html
        response = self.s3_client.generate_presigned_url("get_object", Params={"Bucket": bucket_name, "Key": object_name}, ExpiresIn=expiration)
        return response

    @staticmethod
    def convert_content_length_to_mb(content_length):
        return round(float(f"{content_length / (1e+6):2f}"), 2)

    @explicit_params_validation(validation_type="NoneOrEmpty")
    def copy_key(self, destination_bucket_name: str, source_key: str, destination_key: str):
        self._logger.info(f"Copying the bucket object {self.bucket_name}:{source_key} to {destination_bucket_name}/{destination_key}")
        copy_source = {"Bucket": self.bucket_name, "Key": source_key}

        # Copying the key
        bucket = self.s3_resource.Bucket(destination_bucket_name)
        bucket.copy(copy_source, destination_key)
        return True

    # @explicit_params_validation(validation_type='NoneOrEmpty')
    # def list_common_prefixes(self) -> List[str]:
    #     """
    #     Gets a list of dictionaries, representing directories in the S3 bucket that is passed in the constructor (self.bucket).
    #     :return: The names of the directories in the bucket.
    #     """
    #     paginator = self.s3_client.get_paginator('list_objects_v2')
    #     page_iterator = paginator.paginate(Bucket=self.bucket_name)
    #     prefixes = set()
    #     for item in page_iterator.search('Contents'):
    #         if not item:
    #             continue
    #
    #         if len(item.split('/') == 1):
    #             prefixes.append(item)
    #     return prefixes
