from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from typing import List
import os
from datasets.distributed import split_dataset_by_node


def get_all_tar_files(data_dir: str,
                      num_shards: int) -> List[List[str]]:
    """get path to all .tar files

    Args:
        data_dir (str): path to the data directory
        num_shards (int): number of shards

    Returns:
        List[List[str]]: list of paths to .tar files. Each file path in a sub-list
    """
    all_tar_files = []
    for subdir in os.listdir(data_dir):
        tar_file = os.path.join(data_dir, subdir, f'audio__OP_1..{num_shards-1}_CL_.tar')
        all_tar_files.append([tar_file])
    return all_tar_files


def get_all_manifest_files(data_dir: str) -> List[List[str]]:
    """get path to all .json files

    Args:
        data_dir (str): path to the data directory
        num_shards (int): number of shards

    Returns:
        List[List[str]]: list of paths to .json files. Each file path in a sub-list
    """
    all_manifest_files = []
    for subdir in os.listdir(data_dir):
        manifest_file = os.path.join(data_dir, subdir, 'tarred_audio_manifest.json')
        all_manifest_files.append([manifest_file])
    return all_manifest_files


def load_config(args):
    config = OmegaConf.load(args.config)
    config.model.train_ds.manifest_filepath = get_all_manifest_files(args.data_dir)
    config.model.validation_ds.manifest_filepath = args.valid_manifest
    config.model.test_ds.manifest_filepath = args.test_manifest

    config.model.train_ds.is_tarred = args.is_tarred
    config.model.train_ds.tarred_audio_filepaths = get_all_tar_files(args.data_dir, args.num_shards)

    config.model.tokenizer.dir = args.tokenizer_dir
    config.trainer.max_epochs = args.epochs
    config.trainer.accumulate_grad_batches = args.acc_grad
    config.trainer.precision = args.precision
    config.exp_manager.resume_from_checkpoint = args.resume

    if args.model_size == 'medium':
        config.model.encoder.n_heads = 4
        config.model.encoder.n_layers = 18
        config.model.encoder.d_model = 256
    elif args.model_size == 'large':
        config.model.encoder.n_heads = 8
        config.model.encoder.n_layers = 18
        config.model.decoder.d_model = 512
    elif args.model_size == 'xlarge':
        config.model.encoder.n_heads = 8
        config.model.encoder.n_layers = 24
        config.model.encoder.d_model = 1024
    elif args.model_size == 'small':
        config.model.encoder.n_heads = 4
        config.model.encoder.n_layers = 16
        config.model.encoder.d_model = 176

    return config


def build_model(config):
    model = nemo_asr.models.EncDecCTCModelBPE(cfg=config.model)
    return model

