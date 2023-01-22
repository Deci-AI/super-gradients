import json
import logging

from super_gradients.common.aws_connection.aws_connector import AWSConnector
from super_gradients.common.decorators.explicit_params_validator import explicit_params_validation


class AWSSecretsManagerConnector:
    """
    AWSSecretsManagerConnector - This class handles the AWS Secrets Manager connection
    """

    __slots__ = []  # Making the class immutable for runtime safety
    current_environment_client = None
    DECI_ENVIRONMENTS = ["research", "development", "staging", "production"]

    @staticmethod
    @explicit_params_validation(validation_type="NoneOrEmpty")
    def get_secret_value_for_secret_key(aws_env: str, secret_name: str, secret_key: str) -> str:
        """
        get_secret_value_for_secret_key - Gets a Secret Value from AWS Secrets Manager for the Provided Key
            :param aws_env:         The environment to get the secret for
            :param secret_name: The Secret Name stored in Secrets Manager
            :param secret_key:  The Secret Key To retrieve it's value from AWS
            :return:  str: The Secret Value
        """
        current_class_name = __class__.__name__
        logger = logging.getLogger(current_class_name)
        secret_key = secret_key.upper()
        aws_secrets_dict = AWSSecretsManagerConnector.__get_secrets_manager_dict_for_secret_name(aws_env=aws_env, secret_name=secret_name)

        secret_key = ".".join([aws_env.upper(), secret_key])
        if secret_key not in aws_secrets_dict.keys():
            error = f"[{current_class_name}] - Secret Key ({secret_key}) not Found in AWS Secret: " + secret_name
            logger.error(error)
            raise EnvironmentError(error)
        else:
            return aws_secrets_dict[secret_key]

    @staticmethod
    @explicit_params_validation(validation_type="NoneOrEmpty")
    def get_secret_values_dict_for_secret_key_properties(env: str, secret_key: str, secret_name: str, db_properties_set: set = None) -> dict:
        """
        get_config_dict - Returns the config dict of the properties from the properties dict
            :param  env:                The environment to open the dict for
            :param  secret_key:         The Secret Key
            :param  secret_name:        The Secret to Retrieve to from AWS secrets manager (usually project name)
            :param  db_properties_set:  The set of the properties to get secrets values for
            :return:  dict The secrets dict for the requested property
        """
        current_class_name = __class__.__name__
        logger = logging.getLogger(current_class_name)
        aws_secrets_dict = AWSSecretsManagerConnector.__get_secrets_manager_dict_for_secret_name(aws_env=env, secret_name=secret_name)

        aws_env_safe_secrets = {}
        # FILL THE DICT VALUES FROM THE AWS SECRETS RESPONSE
        if db_properties_set:
            for secret_key_property in db_properties_set:
                secret_key_to_retrieve = ".".join([env.upper(), secret_key, secret_key_property])
                if secret_key_to_retrieve not in aws_secrets_dict:
                    error = (
                        f'[{current_class_name}] - Error retrieving data from AWS Secrets Manager for Secret Key "{secret_name}": '
                        f'The secret property "{secret_key_property}" Does Not Exist'
                    )
                    logger.error(error)
                    raise EnvironmentError(error)
                else:
                    env_stripped_key_name = secret_key_to_retrieve.lstrip(env.upper()).lstrip(".")
                    aws_env_safe_secrets[env_stripped_key_name] = aws_secrets_dict[secret_key_to_retrieve]
        else:
            # "db_properties_set" is not specified - validating and returning all the secret keys and values for
            # the secret name.
            for secret_key_name, secret_value in aws_secrets_dict.items():
                secret_key_to_retrieve = ".".join([env.upper(), secret_key])
                assert secret_key_name.startswith(env.upper()), (
                    f'The secret key property "{secret_key_name}", found in secret named {secret_name}, is not following the convention of '
                    f'environment prefix. please add the environment prefix "{env.upper()}" to property "{secret_key_name}"'
                )
                if secret_key_name.startswith(secret_key_to_retrieve):
                    env_stripped_key_name = secret_key_name.lstrip(env.upper()).lstrip(".")
                    aws_env_safe_secrets[env_stripped_key_name] = secret_value
        return aws_env_safe_secrets

    @staticmethod
    def __get_secrets_manager_dict_for_secret_name(aws_env: str, secret_name: str) -> dict:
        """
        __get_secrets_manager_dict_for_secret_name
            :param  aws_env:                The environment to open the dict for
            :param  secret_name:        The Secret to Retrieve to from AWS secrets manager (usually project name)
            :return: python Dictionary with the key/value pairs stored in AWS Secrets Manager
        """
        current_class_name = __class__.__name__
        logger = logging.getLogger(current_class_name)

        secrets_path = AWSSecretsManagerConnector.__get_secrets_path_from_secret_name(aws_env, secret_name)

        try:
            if not AWSSecretsManagerConnector.current_environment_client:
                logger.debug("Initializing a new secrets manager client...")
                AWSSecretsManagerConnector.current_environment_client = AWSConnector.get_aws_client_for_service_name(
                    profile_name=aws_env, service_name="secretsmanager"
                )
            logger.debug(f'Fetching the secret "{secret_name}" in env "{aws_env}"')
            aws_secrets = AWSSecretsManagerConnector.current_environment_client.get_secret_value(SecretId=secrets_path)
            aws_secrets_dict = json.loads(aws_secrets["SecretString"])
            return aws_secrets_dict

        except Exception as ex:
            error = (
                f'[{current_class_name}] - Caught Exception while trying to connect to aws to get credentials from secrets manager: "{ex}" for {secrets_path}'
            )
            logger.error(error)
            raise EnvironmentError(error)

    @staticmethod
    def __get_secrets_path_from_secret_name(aws_env: str, secret_name: str) -> str:
        """
        __get_secrets_path_from_secret_name - Extracts the full secret path based on the Environment
        :param aws_env:         Env
        :param secret_name: Secret Name
        :return:     str:   The full secret path
        """
        current_class_name = __class__.__name__
        logger = logging.getLogger(current_class_name)

        # Checking for lowercase exact match, in order to prevent any implicit usage of the environments.
        if aws_env not in AWSSecretsManagerConnector.DECI_ENVIRONMENTS:
            logger.critical("[" + current_class_name + " ] -  wrong environment param... Exiting")
            raise Exception("[" + current_class_name + "] - wrong environment param")

        secrets_path = "/".join([aws_env, secret_name])
        return secrets_path
