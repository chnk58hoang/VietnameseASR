from utils import load_config, build_model
import pytorch_lightning as pl
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate manifest file')
    parser.add_argument('--config', type=str, required=True, help='path to the config file')
    parser.add_argument('--data_dir', type=str, required=True, help='path to the tarred data directory')
    parser.add_argument('--valid_manifest', type=str, required=True, help='path to the valid manifest file')
    parser.add_argument('--test_manifest', type=str, required=True, help='path to the test manifest file')
    parser.add_argument('--is_tarred', type=bool, required=False, default=True, help='whether the dataset is tarred')
    parser.add_argument('--tokenizer_dir', type=str, required=True, help='path to the tokenizer .model file')
    parser.add_argument('--model_size', type=str, required=True, default='medium', help='model size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--acc_grad', type=int, required=False, default=2, help='accumulation gradient steps')
    parser.add_argument('--precision', type=int, required=False, choices=[32, 16, 'bf16'] ,default=16,  help='weight precision')
    parser.add_argument('--resume', type=str, required=False, help='resume training from a checkpoint')
    parser.add_argument('--num_shards', type=int, required=False, help='number of shards in the tarred dataset')
    args = parser.parse_args()

    config = load_config(args)
    model = build_model(config)

    trainer = pl.Trainer(**config.trainer)
    trainer.fit(model)
    trainer.test(model)
