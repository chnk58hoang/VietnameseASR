from omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
import argparse


def load_config(args):
    config = OmegaConf.load(args.config)
    config.model.train_ds.manifest_filepath = args.train_manifest
    config.model.validation_ds.manifest_filepath = args.valid_manifest
    config.model.test_ds.manifest_filepath = args.test_manifest

    config.model.train_ds.is_tarred = args.is_tarred
    config.model.train_ds.tarred_path = args.tarred_path

    config.model.tokenizer.dir = args.tokenizer_dir
    config.trainer.max_epochs = args.epochs
    config.trainer.accumulate_grad_batches = args.acc_grad
    config.trainer.precision = args.precision

    config.exp_manager.resume_from_checkpoint = args.resume

    if args.model_size == 'medium':
        config.model.encoder.n_heads = 4
        config.model.encoder.n_layers = 18
        config.model.decoder.d_model = 256
    elif args.model_size == 'large':
        config.model.encoder.n_heads = 8
        config.model.encoder.n_layers = 18
        config.model.decoder.d_model = 512
    elif args.model_size == 'xlarge':
        config.model.encoder.n_heads = 8
        config.model.encoder.n_layers = 24
        config.model.decoder.d_model = 1024
    elif args.model_size == 'small':
        config.model.encoder.n_heads = 4
        config.model.encoder.n_layers = 16
        config.model.decoder.d_model = 176

    return config


def build_model(config):
    model = nemo_asr.models.EncDecCTCModel(cfg=config.model)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate manifest file')
    parser.add_argument('--config', type=str, required=True, help='path to the config file')
    parser.add_argument('--train_manifest', type=str, required=True, help='path to the training manifest file')
    parser.add_argument('--valid_manifest', type=str, required=True, help='path to the valid manifest file')
    parser.add_argument('--test_manifest', type=str, required=True, help='path to the test manifest file')
    parser.add_argument('--is_tarred', type=bool, required=True, help='whether the dataset is tarred')
    parser.add_argument('--tarred_path', type=str, required=True, help='path to the tarred files')
    parser.add_argument('--tokenizer_dir', type=str, required=True, help='path to the tokenizer .model file')
    parser.add_argument('--model_size', type=str, required=True, default='medium', help='model size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--acc_grad', type=int, required=True, default=2, help='accumulation gradient steps')
    parser.add_argument('--precision', type=int, required=True, choices=[32, 16, 'bf16'] ,default=16,  help='weight precision')
    parser.add_argument('--resume', type=str, required=False, help='resume training from a checkpoint')
    args = parser.parse_args()

    config = load_config(args)
    model = build_model(config)

    trainer = pl.Trainer(**config.trainer)
    trainer.fit(model)
    trainer.test(model)
