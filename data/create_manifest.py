from tqdm import tqdm
import argparse
import json
import sox
import os


def create_manifest(data_dir: str,
                    output_file: str = 'manifest.json'):
    """generate data manifest file

    Args:
        data_dir (str): path to data directory
        output_file (str): path to output manifest file. Defaults to 'manifest.json'.
    """
    f = open(output_file, 'w')
    for root, dir, files in os.walk(data_dir):
        for file in tqdm(files, desc='Generating manifest file...', total=len(files)):
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                text_path = file_path.replace('.wav', '.txt')
                f_text = open(text_path, 'r')
                text = f_text.read().strip()
                f_text.close()
                data = {'audio_filepath': file_path,
                        'text': text,
                        "duration": sox.file_info.duration(file_path)}
                f.write(json.dumps(data) + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate manifest file')
    parser.add_argument('--data_dir', type=str, required=True, help='path to data directory')
    parser.add_argument('--output_file', type=str, default='manifest.json', help='path to output manifest file')
    args = parser.parse_args()
    create_manifest(args.data_dir, args.output_file)
