import os
from super_gradients.common import S3Connector
from super_gradients.common import explicit_params_validation
import zipfile


class DatasetDataInterface:

    def __init__(self, env: str, data_connection_source: str = 's3'):
        """

        :param env: str "development"/"production"
        :param data_connection_source: str "s3" for aws bny default
        """
        self.env = env
        self.s3_connector = None
        self.data_connection_source = data_connection_source

    @explicit_params_validation(validation_type='None')
    def load_remote_dataset_file(self, remote_file: str, local_dir: str, overwrite_local_dataset: bool = False) -> str:
        """

        :param remote_file: str - the name of s3 file
        :param local_dir: str - the directory to put the dataset in
        :param overwrite_local_dataset: Whether too  delete the dataset dir before downloading
        :return:
        """

        dataset_full_path = local_dir
        bucket = remote_file.split("/")[2]
        file_path = "/".join(remote_file.split("/")[3:])
        if self.data_connection_source == 's3':
            self.s3_connector = S3Connector(self.env, bucket)

            # DELETE THE LOCAL VERSION ON THE MACHINE
            if os.path.exists(dataset_full_path):
                if overwrite_local_dataset:

                    filelist = os.listdir(local_dir)
                    for f in filelist:
                        os.remove(os.path.join(local_dir, f))
                else:
                    Warning("Overwrite local dataset set to False but dataset exists in the dir")
            if not os.path.exists(local_dir):
                os.mkdir(local_dir)

            local_file = self.s3_connector.download_file_by_path(file_path, local_dir)
            with zipfile.ZipFile(local_dir + "/" + local_file, 'r') as zip_ref:
                zip_ref.extractall(local_dir + "/")
            os.remove(local_dir + "/" + local_file)

        return local_dir
