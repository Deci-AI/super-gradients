import os
import sys
from super_gradients.common import S3Connector, explicit_params_validation
from super_gradients.common.abstractions.abstract_logger import ILogger


class ModelCheckpointNotFoundException(RuntimeError):
    pass


class ADNNModelRepositoryDataInterfaces(ILogger):
    """
    ResearchModelRepositoryDataInterface
    """

    def __init__(self, data_connection_location: str = 'local', data_connection_credentials: str = None):
        """
        ModelCheckpointsDataInterface
            :param data_connection_location: 'local' or s3 bucket 's3://my-bucket-name'
            :param data_connection_credentials: credentials string
                    - name of aws profile in case data_connection_source is s3. will be taken form environment variable
                    AWS_PROFILE if left empty
        """
        super().__init__()
        self.tb_events_file_prefix = 'events.out.tfevents'
        self.log_file_prefix = 'log_'
        self.latest_checkpoint_filename = 'ckpt_latest.pth'
        self.best_checkpoint_filename = 'ckpt_best.pth'

        if data_connection_location.startswith('s3'):
            assert data_connection_location.index('s3://') >= 0, 'S3 path must be formatted s3://bucket-name'
            self.model_repo_bucket_name = data_connection_location.split('://')[1]
            self.data_connection_source = 's3'

            if data_connection_credentials is None:
                data_connection_credentials = os.getenv('AWS_PROFILE')

            self.s3_connector = S3Connector(data_connection_credentials, self.model_repo_bucket_name)

    @explicit_params_validation(validation_type='None')
    def load_all_remote_log_files(self, model_name: str, model_checkpoint_local_dir: str):
        """
        load_all_remote_checkpoint_files
            :param model_name:
            :param model_checkpoint_local_dir:
            :return:
        """
        self.load_remote_logging_files(model_name=model_name, model_checkpoint_dir_name=model_checkpoint_local_dir,
                                       logging_type='tensorboard')
        self.load_remote_logging_files(model_name=model_name, model_checkpoint_dir_name=model_checkpoint_local_dir,
                                       logging_type='text')

    @explicit_params_validation(validation_type='None')
    def save_all_remote_checkpoint_files(self, model_name: str, model_checkpoint_local_dir: str, log_file_name: str):
        """
        save_all_remote_checkpoint_files - Saves all of the local Checkpoint data into Remote Repo
            :param model_name:                  The Model Name to store in Remote Repo
            :param model_checkpoint_local_dir:  Local directory with the relevant data to upload
            :param log_file_name:               The log_file name (Created independently)
        """
        for checkpoint_file_name in [self.latest_checkpoint_filename, self.best_checkpoint_filename]:
            self.save_remote_checkpoints_file(model_name, model_checkpoint_local_dir, checkpoint_file_name)

        self.save_remote_checkpoints_file(model_name, model_checkpoint_local_dir, log_file_name)
        self.save_remote_tensorboard_event_files(model_name, model_checkpoint_local_dir)

    @explicit_params_validation(validation_type='None')
    def load_remote_checkpoints_file(self, ckpt_source_remote_dir: str, ckpt_destination_local_dir: str,
                                     ckpt_file_name: str, overwrite_local_checkpoints_file: bool = False) -> str:
        """
        load_remote_checkpoints_file - Loads a model's checkpoint from local/cloud file
            :param ckpt_source_remote_dir:               The source folder to download from
            :param ckpt_destination_local_dir:           The destination folder to save the checkpoint at
            :param ckpt_file_name:                       Filename to load from Remote Repo
            :param overwrite_local_checkpoints_file:     Use Only for Cloud-Stored Model Checkpoints if required behavior
                                                            is to overwrite a previous version of the same files
            :return: Model Checkpoint File Path -> Depends on model architecture
        """
        ckpt_file_local_full_path = ckpt_destination_local_dir + '/' + ckpt_file_name

        if self.data_connection_source == 's3':
            if overwrite_local_checkpoints_file:
                # DELETE THE LOCAL VERSION ON THE MACHINE
                if os.path.exists(ckpt_file_local_full_path):
                    os.remove(ckpt_file_local_full_path)

            key_to_download = ckpt_source_remote_dir + '/' + ckpt_file_name
            download_success = self.s3_connector.download_key(target_path=ckpt_file_local_full_path,
                                                              key_to_download=key_to_download)

            if not download_success:
                failed_download_path = 's3://' + self.model_repo_bucket_name + '/' + key_to_download
                error_msg = 'Failed to Download Model Checkpoint from ' + failed_download_path
                self._logger.error(error_msg)
                raise ModelCheckpointNotFoundException(error_msg)

        return ckpt_file_local_full_path

    @explicit_params_validation(validation_type='NoneOrEmpty')
    def load_remote_logging_files(self, model_name: str, model_checkpoint_dir_name: str, logging_type: str):
        """
        load_remote_tensorboard_event_files - Downloads all of the tb_events Files from remote repository
            :param model_name:
            :param model_checkpoint_dir_name:
            :param logging_type:
            :return:
        """
        if not os.path.isdir(model_checkpoint_dir_name):
            raise ValueError('[' + sys._getframe().f_code.co_name + '] - Provided directory does not exist')

        # LOADS THE DATA FROM THE REMOTE REPOSITORY
        s3_bucket_path_prefix = model_name
        if logging_type == 'tensorboard':
            if self.data_connection_source == 's3':
                self.s3_connector.download_keys_by_prefix(s3_bucket_path_prefix=s3_bucket_path_prefix,
                                                          local_download_dir=model_checkpoint_dir_name,
                                                          s3_file_path_prefix=self.tb_events_file_prefix)
        elif logging_type == 'text':
            if self.data_connection_source == 's3':
                self.s3_connector.download_keys_by_prefix(s3_bucket_path_prefix=s3_bucket_path_prefix,
                                                          local_download_dir=model_checkpoint_dir_name,
                                                          s3_file_path_prefix=self.log_file_prefix)

    @explicit_params_validation(validation_type='NoneOrEmpty')
    def save_remote_checkpoints_file(self, model_name: str, model_checkpoint_local_dir: str,
                                     checkpoints_file_name: str) -> bool:
        """
        save_remote_checkpoints_file - Saves a Checkpoints file in the Remote Repo
            :param model_name:                      The Model Name for S3 Prefix
            :param model_checkpoint_local_dir:      Model Directory - Based on Model name
            :param checkpoints_file_name:           Filename to upload to Remote Repo
            :return: True/False for Operation Success/Failure
        """
        # LOAD THE LOCAL VERSION
        model_checkpoint_file_full_path = model_checkpoint_local_dir + '/' + checkpoints_file_name

        # SAVE ON THE REMOTE S3 REPOSITORY
        if self.data_connection_source == 's3':
            model_checkpoint_s3_in_bucket_path = model_name + '/' + checkpoints_file_name
            return self.__update_or_upload_s3_key(model_checkpoint_file_full_path, model_checkpoint_s3_in_bucket_path)

    @explicit_params_validation(validation_type='NoneOrEmpty')
    def save_remote_tensorboard_event_files(self, model_name: str, model_checkpoint_dir_name: str):
        """
        save_remote_tensorboard_event_files - Saves all of the tensorboard files remotely
            :param model_name:                Prefix for Cloud Storage
            :param model_checkpoint_dir_name: The directory where the files are stored in
        """
        if not os.path.isdir(model_checkpoint_dir_name):
            raise ValueError('[' + sys._getframe().f_code.co_name + '] - Provided directory does not exist')

        for tb_events_file_name in os.listdir(model_checkpoint_dir_name):
            if tb_events_file_name.startswith(self.tb_events_file_prefix):
                upload_success = self.save_remote_checkpoints_file(model_name=model_name,
                                                                   model_checkpoint_local_dir=model_checkpoint_dir_name,
                                                                   checkpoints_file_name=tb_events_file_name)

                if not upload_success:
                    self._logger.error('Failed to upload tb_events_file: ' + tb_events_file_name)

    @explicit_params_validation(validation_type='NoneOrEmpty')
    def __update_or_upload_s3_key(self, local_file_path: str, s3_key_path: str):
        """
        __update_or_upload_s3_key - Uploads/Updates an S3 Key based on a local file path
            :param local_file_path: The Local file path to upload to S3
            :param s3_key_path:     The S3 path to create/update the S3 Key
        """
        if self.s3_connector.check_key_exists(s3_key_path):
            # DELETE KEY TO UPDATE THE FILE IN S3
            delete_response = self.s3_connector.delete_key(s3_key_path)
            if delete_response:
                self._logger.info('Removed previous checkpoint from S3')

        upload_success = self.s3_connector.upload_file(local_file_path, s3_key_path)
        if not upload_success:
            self._logger.error('Failed to upload model checkpoint')

        return upload_success
