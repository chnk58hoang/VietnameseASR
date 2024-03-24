import os
import librosa
import argparse


def resample_wav_files(directory, origin_sr, target_sr):
    for root, dir, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".wav"):
                filepath = os.path.join(directory, filename)
                y, sr = librosa.load(filepath, sr=origin_sr)
                y_resampled = librosa.resample(y, sr, target_sr)
                librosa.output.write_wav(filepath, y_resampled, target_sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resample wav files')
    parser.add_argument('--directory', type=str, required=True, help='path to data directory')
    parser.add_argument('--origin_sr', type=int, required=True, help='original sampling rate')
    parser.add_argument('--target_sr', type=int, required=True, help='target sampling rate')
    args = parser.parse_args()
    resample_wav_files(args.directory, args.origin_sr, args.target_sr)
