import json
import os
import math
import librosa
from collections import Counter
import pandas as pd
import numpy as np

stp = "/Users/andrea/Documents/DHDK/Thesis/MIDI/prediction"
DATASET_PATH = "/Users/andrea/Documents/DHDK/Thesis/Music Emotion/MER_audio_taffc_dataset"
CSV_PATH = "/Users/andrea/Documents/DHDK/Thesis/Music Emotion/set1/mean_ratings_set1.xls"
JSON_PATH = "/Users/andrea/Documents/DHDK/Thesis/Music Emotion/mfccs_5.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 29.7  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10, mode='onlymfcc'):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
    """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:
                print(f)
                ext = os.path.splitext(f)[-1].lower()
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                dur = librosa.core.get_duration(signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
                if ext == '.mp3' and dur > 29:

                    # load audio file

                    # process all segments of audio file
                    for d in range(num_segments):
                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        if mode is 'onlymfcc':
                            # extract mfcc
                            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                        hop_length=hop_length)
                            mfcc = mfcc.T

                        elif mode is 'allfeatures':

                            mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                                                hop_length=hop_length)
                            mfcc = mfcc.T

                            spectral_center = librosa.feature.spectral_centroid(signal[start:finish], sample_rate,
                                                                                hop_length=hop_length)

                            spectral_center = spectral_center.T

                            chroma = librosa.feature.chroma_stft(signal[start:finish], sample_rate, n_chroma=num_mfcc,
                                                                                hop_length=hop_length)

                            chroma = chroma.T

                            spectral_contrast = librosa.feature.spectral_contrast(signal[start:finish], sample_rate,
                                                                                  hop_length=hop_length)

                            spectral_contrast = spectral_contrast.T

                            # initialize an empty np.array
                            arr = np.zeros((len(mfcc), 34))

                            # populate the array
                            arr[:, 0:13] = mfcc
                            arr[:, 13:14] = spectral_center
                            arr[:, 14:27] = chroma
                            arr[:, 27:34] = spectral_contrast
                            data["mfcc"].append(arr.tolist())

                            # store only mfcc feature with expected number of vectors
                            if len(mfcc) == num_mfcc_vectors_per_segment:
                                if mode is 'onlymfcc':
                                    data["mfcc"].append(mfcc.tolist())
                                if mode is 'allfeatures':
                                    data["mfcc"].append(data.tolist())
                                data["labels"].append(i - 1)
                                print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def retrieve_labels(dataset_path):
    genre_list = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            directory, genre = os.path.split(dirpath)
            genre_list.append(genre)
    print(genre_list)


if __name__ == "__main__":
    save_mfcc(stp, JSON_PATH, num_segments=10, mode='onlymfcc')
    # retrieve_labels(DATASET_PATH)
