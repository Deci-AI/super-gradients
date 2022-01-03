import os
import tempfile
import pkg_resources
import torch
from super_gradients.common import explicit_params_validation, ADNNModelRepositoryDataInterfaces
from super_gradients.training.pretrained_models import MODEL_URLS
try:
    from torch.hub import download_url_to_file, load_state_dict_from_url
except (ModuleNotFoundError, ImportError, NameError):
    from torch.hub import _download_url_to_file as download_url_to_file


def get_ckpt_local_path(source_ckpt_folder_name: str, experiment_name: str, ckpt_name: str, model_checkpoints_location: str, external_checkpoint_path: str, overwrite_local_checkpoint: bool, load_weights_only: bool):
    """
    Gets the local path to the checkpoint file, which will be:
        - By default: YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name.
        - if the checkpoint file is remotely located:
            when overwrite_local_checkpoint=True then it will be saved in a temporary path which will be returned,
            otherwise it will be downloaded to YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name and overwrite
            YOUR_REPO_ROOT/super_gradients/checkpoints/experiment_name/ckpt_name if such file exists.
        - external_checkpoint_path when external_checkpoint_path != None

    @param source_ckpt_folder_name: The folder where the checkpoint is saved. When set to None- uses the experiment_name.
    @param experiment_name: experiment name attr in sg_model
    @param ckpt_name: checkpoint filename
    @param model_checkpoints_location: S3, local ot URL
    @param external_checkpoint_path: full path to checkpoint file (that might be located outside of super_gradients/checkpoints directory)
    @param overwrite_local_checkpoint: whether to overwrite the checkpoint file with the same name when downloading from S3.
    @param load_weights_only: whether to load the network's state dict only.
    @return:
    """
    source_ckpt_folder_name = source_ckpt_folder_name or experiment_name
    if model_checkpoints_location == 'local':
        default_local_ckpt_path = pkg_resources.resource_filename('checkpoints',
                                                                  source_ckpt_folder_name + os.path.sep + ckpt_name)
        ckpt_local_path = external_checkpoint_path or default_local_ckpt_path

    # COPY THE DATA FROM 'S3'/'URL' INTO A LOCAL DIRECTORY
    elif model_checkpoints_location.startswith('s3') or model_checkpoints_location == 'url':
        # COPY REMOTE DATA TO A LOCAL DIRECTORY AND GET THAT DIRECTORYs NAME
        ckpt_local_path = copy_ckpt_to_local_folder(local_ckpt_destination_dir=experiment_name,
                                                    ckpt_filename=ckpt_name,
                                                    remote_ckpt_source_dir=source_ckpt_folder_name,
                                                    path_src=model_checkpoints_location,
                                                    overwrite_local_ckpt=overwrite_local_checkpoint,
                                                    load_weights_only=load_weights_only)

    else:
        # ERROR IN USER CODE FLOW - THIS WILL EVENTUALLY RAISE AN EXCEPTION
        raise NotImplementedError(
            'model_checkpoints_data_source: ' + str(model_checkpoints_location) + 'not supported')

    return ckpt_local_path


def adaptive_load_state_dict(net: torch.nn.Module, state_dict: dict, strict: str):
    """
    Adaptively loads state_dict to net, by adapting the state_dict to net's layer names first.

    @param net: (nn.Module) to load state_dict to
    @param state_dict: (dict) Chekpoint state_dict
    @param strict: (str) key matching strictness
    @return:
    """
    try:
        net.load_state_dict(state_dict['net'], strict=strict)
    except (RuntimeError, ValueError, KeyError) as ex:
        if strict == 'no_key_matching':
            adapted_state_dict = adapt_state_dict_to_fit_model_layer_names(net.state_dict(), state_dict)
            net.load_state_dict(adapted_state_dict['net'], strict=True)
        else:
            raise_informative_runtime_error(net.state_dict(), state_dict, ex)


@explicit_params_validation(validation_type='None')
def copy_ckpt_to_local_folder(local_ckpt_destination_dir: str, ckpt_filename: str, remote_ckpt_source_dir: str = None,
                              path_src: str = 'local', overwrite_local_ckpt: bool = False,
                              load_weights_only: bool = False):
    """
    Copy the checkpoint from any supported source to a local destination path
        :param local_ckpt_destination_dir:  destination where the checkpoint will be saved to
        :param ckpt_filename:         ckpt_best.pth Or ckpt_latest.pth
        :param remote_ckpt_source_dir:       Name of the source checkpoint to be loaded (S3 Model\full URL)
        :param path_src:              S3 / url
        :param overwrite_local_ckpt:  determines if checkpoint will be saved in destination dir or in a temp folder

        :return: Path to checkpoint
    """
    ckpt_file_full_local_path = None

    # IF NOT DEFINED - IT IS SET TO THE TARGET's FOLDER NAME
    remote_ckpt_source_dir = local_ckpt_destination_dir if remote_ckpt_source_dir is None else remote_ckpt_source_dir

    if not overwrite_local_ckpt:
        # CREATE A TEMP FOLDER TO SAVE THE CHECKPOINT TO
        download_ckpt_destination_dir = tempfile.gettempdir()
        print('PLEASE NOTICE - YOU ARE IMPORTING A REMOTE CHECKPOINT WITH overwrite_local_checkpoint = False '
              '-> IT WILL BE REDIRECTED TO A TEMP FOLDER AND DELETED ON MACHINE RESTART')
    else:
        # SAVE THE CHECKPOINT TO MODEL's FOLDER
        download_ckpt_destination_dir = pkg_resources.resource_filename('checkpoints', local_ckpt_destination_dir)

    if path_src.startswith('s3'):
        model_checkpoints_data_interface = ADNNModelRepositoryDataInterfaces(data_connection_location=path_src)
        # DOWNLOAD THE FILE FROM S3 TO THE DESTINATION FOLDER
        ckpt_file_full_local_path = model_checkpoints_data_interface.load_remote_checkpoints_file(
            ckpt_source_remote_dir=remote_ckpt_source_dir,
            ckpt_destination_local_dir=download_ckpt_destination_dir,
            ckpt_file_name=ckpt_filename,
            overwrite_local_checkpoints_file=overwrite_local_ckpt)

        if not load_weights_only:
            # COPY LOG FILES FROM THE REMOTE DIRECTORY TO THE LOCAL ONE ONLY IF LOADING THE CURRENT MODELs CKPT
            model_checkpoints_data_interface.load_all_remote_log_files(model_name=remote_ckpt_source_dir,
                                                                       model_checkpoint_local_dir=download_ckpt_destination_dir)

    if path_src == 'url':
        ckpt_file_full_local_path = download_ckpt_destination_dir + os.path.sep + ckpt_filename
        # DOWNLOAD THE FILE FROM URL TO THE DESTINATION FOLDER
        download_url_to_file(remote_ckpt_source_dir, ckpt_file_full_local_path, progress=True)

    return ckpt_file_full_local_path


def read_ckpt_state_dict(ckpt_path: str, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise ValueError('Incorrect Checkpoint path')

    if device == "cuda":
        state_dict = torch.load(ckpt_path)

    else:
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    return state_dict


def adapt_state_dict_to_fit_model_layer_names(model_state_dict: dict, source_ckpt: dict, exclude: list = []):
    """
    Given a model state dict and source checkpoints, the method tries to correct the keys in the model_state_dict to fit
    the ckpt in order to properly load the weights into the model. If unsuccessful - returns None
        :param model_state_dict:               the model state_dict
        :param source_ckpt:                         checkpoint dict
        :exclude                  optional list for excluded layers
        :return: renamed checkpoint dict (if possible)
    """
    if 'net' in source_ckpt.keys():
        source_ckpt = source_ckpt['net']
    model_state_dict_excluded = {k: v for k, v in model_state_dict.items() if not any(x in k for x in exclude)}
    new_ckpt_dict = {}
    for (ckpt_key, ckpt_val), (model_key, model_val) in zip(source_ckpt.items(), model_state_dict_excluded.items()):
        if ckpt_val.shape != model_val.shape:
            raise ValueError(f'ckpt layer {ckpt_key} with shape {ckpt_val.shape} does not match {model_key}'
                             f' with shape {model_val.shape} in the model')
        new_ckpt_dict[model_key] = ckpt_val
    return {'net': new_ckpt_dict}


def raise_informative_runtime_error(state_dict, checkpoint, exception_msg):
    """
    Given a model state dict and source checkpoints, the method calls "adapt_state_dict_to_fit_model_layer_names"
    and enhances the exception_msg if loading the checkpoint_dict via the conversion method is possible
    """
    try:
        new_ckpt_dict = adapt_state_dict_to_fit_model_layer_names(state_dict, checkpoint)
        temp_file = tempfile.NamedTemporaryFile().name + '.pt'
        torch.save(new_ckpt_dict, temp_file)
        exception_msg = f"\n{'=' * 200}\n{str(exception_msg)} \nconvert ckpt via the utils.adapt_state_dict_to_fit_" \
                        f"model_layer_names method\na converted checkpoint file was saved in the path {temp_file}\n{'=' * 200}"
    except ValueError as ex:  # IN CASE adapt_state_dict_to_fit_model_layer_names WAS UNSUCCESSFUL
        exception_msg = f"\n{'=' * 200} \nThe checkpoint and model shapes do no fit, e.g.: {ex}\n{'=' * 200}"
    finally:
        raise RuntimeError(exception_msg)


def load_checkpoint_to_model(ckpt_local_path: str, load_backbone: bool, net: torch.nn.Module, strict: str, load_weights_only: bool, load_ema_as_net: bool = False):
    """
    Loads the state dict in ckpt_local_path to net and returns the checkpoint's state dict.

    @param load_ema_as_net: Will load the EMA inside the checkpoint file to the network when set
    @param ckpt_local_path: local path to the checkpoint file
    @param load_backbone: whether to load the checkpoint as a backbone
    @param net: network to load the checkpoint to
    @param strict:
    @param load_weights_only:
    @return:
    """
    if ckpt_local_path is None or not os.path.exists(ckpt_local_path):
        error_msg = 'Error - loading Model Checkpoint: Path {} does not exist'.format(ckpt_local_path)
        raise RuntimeError(error_msg)

    if load_backbone and not hasattr(net.module, 'backbone'):
        raise ValueError("No backbone attribute in net - Can't load backbone weights")

    # LOAD THE LOCAL CHECKPOINT PATH INTO A state_dict OBJECT
    checkpoint = read_ckpt_state_dict(ckpt_path=ckpt_local_path)

    if load_ema_as_net:
        if 'ema_net' not in checkpoint.keys():
            raise ValueError("Can't load ema network- no EMA network stored in checkpoint file")
        else:
            checkpoint['net'] = checkpoint['ema_net']

    # LOAD THE CHECKPOINTS WEIGHTS TO THE MODEL
    if load_backbone:
        adaptive_load_state_dict(net.module.backbone, checkpoint, strict)
    else:
        adaptive_load_state_dict(net, checkpoint, strict)

    if load_weights_only or load_backbone:
        # DISCARD ALL THE DATA STORED IN CHECKPOINT OTHER THAN THE WEIGHTS
        [checkpoint.pop(key) for key in list(checkpoint.keys()) if key != 'net']

    return checkpoint


class MissingPretrainedWeightsException(Exception):
    """Exception raised by unsupported pretrianed model.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, desc):
        self.message = "Missing pretrained wights: " + desc
        super().__init__(self.message)


def load_pretrained_weights(model: torch.nn.Module, architecture: str, pretrained_weights: str):

    """
    Loads pretrained weights from the MODEL_URLS dictionary to model
    @param architecture: name of the model's architecture
    @param model: model to load pretrinaed weights for
    @param pretrained_weights: name for the pretrianed weights (i.e imagenet)
    @return: None
    """
    model_url_key = architecture + '_' + str(pretrained_weights)
    if model_url_key not in MODEL_URLS.keys():
        raise MissingPretrainedWeightsException(model_url_key)

    url = MODEL_URLS[model_url_key]
    unique_filename = url.split("https://deci-pretrained-models.s3.amazonaws.com/")[1].replace('/', '_').replace(' ', '_')
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    pretrained_state_dict = load_state_dict_from_url(url=url, map_location=map_location, file_name=unique_filename)
    if 'ema_net' in pretrained_state_dict.keys():
        pretrained_state_dict['net'] = pretrained_state_dict['ema_net']
    adapted_pretrained_state_dict = adapt_state_dict_to_fit_model_layer_names(model_state_dict=model.state_dict(), source_ckpt=pretrained_state_dict)
    model.load_state_dict(adapted_pretrained_state_dict['net'], strict=False)
