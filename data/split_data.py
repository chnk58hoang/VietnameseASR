from random import shuffle
import json
import argparse


def write_manifest(data, output_file):
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def split_data(manifest: str,
               train_ratio: float = 0.7,
               val_ratio: float = 0.1):
    """split data into train, validation and test sets
    Args:
        manifest (str): path to manifest file
        train_ratio (float): ratio of training data. Defaults to 0.7.
        val_ratio (float): ratio of validation data. Defaults to 0.1.
        test_ratio (float): ratio of test data. Defaults to 0.2.
    """
    with open(manifest, 'r') as f:
        data = [json.loads(line) for line in f]
    f.close()
    group1 = list(filter(lambda x: x['duration'] <= 5, data))
    group2 = list(filter(lambda x: 5 < x['duration'] <= 10, data))
    group3 = list(filter(lambda x: 10 < x['duration'] <= 20, data))
    group4 = list(filter(lambda x: 20 < x['duration'] <= 30, data))
    group5 = list(filter(lambda x: x['duration'] > 30, data))
    train_manifest = []
    valid_manifest = []
    test_manifest = []
    for group in [group1, group2, group3, group4, group5]:
        shuffle(group)
        train_index = int(len(group) * train_ratio)
        val_index = int(len(group) * (train_ratio + val_ratio))
        train_set = group[:train_index]
        val_set = group[train_index:val_index]
        test_set = group[val_index:]
        train_manifest.extend(train_set)
        valid_manifest.extend(val_set)
        test_manifest.extend(test_set)
    return train_manifest, valid_manifest, test_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split manifest file')
    parser.add_argument('--manifest', type=str, required=True, help='path to manifest file')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')
    args = parser.parse_args()
    train, valid, test = split_data(args.manifest)
    write_manifest(train, args.output_dir + '/train_manifest.json')
    write_manifest(valid, args.output_dir + '/valid_manifest.json')
    write_manifest(test, args.output_dir + '/test_manifest.json')
