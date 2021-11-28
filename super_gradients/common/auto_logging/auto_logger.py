import json
import os

import pkg_resources

from super_gradients.common.aws_connection.aws_secrets_manager_connector import AWSSecretsManagerConnector
from super_gradients.common.environment import AWS_ENV_NAME
from super_gradients.common.environment.environment_config import DONT_USE_ELASTICSEARCH_LOGGER, DEFAULT_LOGGING_LEVEL


class AutoLoggerConfig:
    """
    A Class for the Automated Logging Config that is created from the JSON config file (auto_logging_conf)
    """

    @staticmethod
    def generate_config_for_module_name(module_name, training_log_path=None, log_level=DEFAULT_LOGGING_LEVEL, max_bytes=10485760, logs_dir_path=None,
                                        handlers_list=None) -> dict:
        """
        generate_config_for_module_name - Returns a Config Dict For Logging
            :param module_name:     The Python Module name to create auto_logging for
            :param log_level:       Minimal log level to set for the new auto_logging
            :param max_bytes:       Max size for the log file before rotation starts
            :param handlers_list:    A list specifying the handlers (Console, etc..) - Better Leave Empty or None
            :param training_log_path: Path to training log file which all modules of super_gradients will write to. Ignored
             when set to None.
            :param logs_dir_path: Path to sg_logs directory (default=None), where module logs will be saved. When set
                to None- module logs will be saved in ~/sg_logs (created if path does not exist). Main use case is for
                testing.


            :return: python dict() with the new auto_logging for the module
        """

        # LOADING THE ORIGINAL ROOT CONFIG FILE
        conf_file_name = 'auto_logging_conf.json'
        conf_file_path = os.path.join(pkg_resources.resource_filename('super_gradients', '/common/auto_logging/'),
                                      conf_file_name)

        with open(conf_file_path, 'r') as logging_configuration_file:
            config_dict = json.load(logging_configuration_file)

        # CREATING THE PATH TO THE "HOME" FOLDER WITH THE LOG FILE NAME
        if not logs_dir_path:
            log_file_name = module_name + '.log'
            user_dir = os.path.expanduser(r"~")
            logs_dir_path = os.path.join(user_dir, 'sg_logs')

        if not os.path.exists(logs_dir_path):
            try:
                os.mkdir(logs_dir_path)
            except Exception as ex:
                print('[WARNING] - sg_logs folder was not found and couldn\'t be created from the code - '
                      'All of the Log output will be sent to Console!' + str(ex))

            # HANDLERS LIST IS EMPTY AS CONSOLE IS ONLY ROOT HANDLER BECAUSE MODULE LOGGERS PROPAGATE THEIR LOGS UP.
            handlers_list = []
            logger = {
                "level": log_level,
                "handlers": handlers_list,
                "propagate": True
            }
            config_dict['loggers'][module_name] = logger

            return config_dict

        log_file_path = os.path.join(logs_dir_path, log_file_name)

        # THE ENTRIES TO ADD TO THE ORIGINAL CONFIGURATION
        handler_name = module_name + '_file_handler'
        file_handler = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "fileFormatter",
            "filename": log_file_path,
            "maxBytes": max_bytes,
            "backupCount": 20,
            "encoding": "utf8"
        }

        # CREATING ONLY A FILE HANDLER, CONSOLE IS ONLY ROOT HANDLER AS MODULE LOGGERS PROPAGATE THEIR LOGS UP.
        if handlers_list is None or handlers_list.empty():
            handlers_list = [handler_name]

        logger = {
            "level": log_level,
            "handlers": handlers_list,
            "propagate": True
        }

        # ADDING THE NEW LOGGER ENTRIES TO THE CONFIG DICT
        config_dict['handlers'][handler_name] = file_handler
        config_dict['loggers'][module_name] = logger
        config_dict['root']['handlers'].append(handler_name)

        if DONT_USE_ELASTICSEARCH_LOGGER:
            return config_dict

        # Creating a ElasticSearch handler
        elastic_handler, elastic_handler_name = AutoLoggerConfig.configure_elasticsearch_handler(config_dict,
                                                                                                 module_name)
        if elastic_handler and elastic_handler_name:
            handlers_list.append(elastic_handler_name)
            config_dict['handlers'][elastic_handler_name] = elastic_handler

        if training_log_path:
            training_file_handler = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "fileFormatter",
                "filename": training_log_path,
                "maxBytes": max_bytes,
                "backupCount": 20,
                "encoding": "utf8"
            }

            # ALL OF DECI_TRAINER MODULES LOGGERS PROPAGATE UP TO THE ROOT SO THE ADD TRAIN FILE HANDLER FOR THE ROOT.
            config_dict['handlers']["training"] = training_file_handler
            config_dict['root']['handlers'].append("training")

        return config_dict

    @staticmethod
    def configure_elasticsearch_handler(config_dict: dict, module_name: str):
        """
        Configures the ElasticSearch loggeing handler through an matching library.
        """
        # Getting the elasticsearch secrets
        if not AWS_ENV_NAME:
            return None, None

        try:
            elastic_secrets = AWSSecretsManagerConnector. \
                get_secret_values_dict_for_secret_key_properties(env=AWS_ENV_NAME,
                                                                 secret_name='elasticLogging',
                                                                 secret_key='ELASTIC')

            # logging_user_name = elastic_secrets['ELASTIC.USERNAME']
            # logging_user_password = elastic_secrets['ELASTIC.PASSWORD']
            elastic_host = elastic_secrets['ELASTIC.HOST']
            elastic_port = int(elastic_secrets['ELASTIC.PORT'])
            elastic_index_name = elastic_secrets['ELASTIC.DEFAULT_INDEX_NAME']
            flush_frequency = int(elastic_secrets['ELASTIC.FLUSH_FREQUENCY_SECONDS'])

            # We import here because not everybody may want elasticsearch handler, thus doesn't need CMRESHandler library.
            from cmreslogging.handlers import CMRESHandler
            config_dict['handlers']['elasticsearch'] = {
                "level": "DEBUG",
                "class": "cmreslogging.handlers.CMRESHandler",
                "hosts": [
                    {
                        "host": elastic_host,
                        "port": elastic_port
                    }
                ],
                "es_index_name": elastic_index_name,
                "es_additional_fields": {
                    "App": "Deci",
                    "Environment": AWS_ENV_NAME
                },
                "auth_type": CMRESHandler.AuthType.NO_AUTH,
                # "auth_details": [
                #     logging_user_name,
                #     logging_user_password
                # ],
                "use_ssl": True,
                "flush_frequency_in_sec": flush_frequency
            }
            elastic_handler = config_dict['handlers']['elasticsearch']
            elastic_handler_name = module_name + '_elastic_handler'
            return elastic_handler, elastic_handler_name
        except Exception as e:
            print(f'Failed to get the elasticsearch logging secrets: {e}')
            return None, None
