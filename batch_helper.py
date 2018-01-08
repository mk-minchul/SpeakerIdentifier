from random import shuffle
import librosa
import numpy as np
import os


def good(filename):
    # returns true if the filename has  name and the wav extension.
    return len(os.path.basename(filename).split("-")) == 3 and os.path.basename(filename).rsplit(".")[-1] == "wav"


def speaker_of(filename):
    # get the speaker in the file name
    return filename.split("-")[0]


def get_speakers(path):
    files = os.listdir(path)

    def nobad(name):
        return "_" in name and not "." in name.split("_")[1]

    speakers = list(set(map(speaker_of, filter(good, files))))
    print(len(speakers), " speakers: ", speakers)
    speakers.sort()
    return speakers

def one_hot_from_item(item, items):
    x = [0] * len(items)  # numpy.zeros(len(items))
    i = items.index(item)
    x[i] = 1
    return x

def mfcc_batch_generator(batch_size, path, max_filesize):
    speakers = get_speakers(path)  # unique speakers list
    batch_features = []  # its like X
    labels = []  # its like Y
    all_files = os.listdir(path)
    print("loaded {} files before capping max_filesize of {}".format(len(all_files), max_filesize))

    # remove wav files bigger than max_filesize and get the biggest file path
    biggest_file_path = ""
    biggest_file_size = 0
    files = []
    for wav in all_files:
        wav_path = os.path.join(path, wav)
        if os.path.getsize(wav_path) < max_filesize:
            files.append(wav)
            if os.path.getsize(wav_path) > biggest_file_size:
                biggest_file_path = wav_path
                biggest_file_size = os.path.getsize(wav_path)

    print("loaded {} files after capping max_filesize of {}".format(len(files), max_filesize))

    # get the biggest file's frame count and set it as the max pad length
    wave, sr = librosa.load(biggest_file_path, mono=True)
    biggest_mfcc = librosa.feature.mfcc(wave, sr)
    print("The biggest file's frame count is ", biggest_mfcc.shape[1])
    pad_width = biggest_mfcc.shape[1]

    while True:
        shuffle(files)
        for wav in files:

            if not wav.endswith(".wav"): continue

            wav_path = os.path.join(path, wav)
            wave, sr = librosa.load(wav_path, mono=True)
            label = one_hot_from_item(speaker_of(wav), speakers)
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width - len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                # print(np.array(batch_features).shape)
                # yield np.array(batch_features), labels
                yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch
                labels = []

def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)


def mfcc_test_X_generator(batch_size, path, max_filesize, num_features):
    # used for testing
    batch_features = []  # its like X
    all_files = os.listdir(path)
    print("loaded {} files before capping max_filesize of {}".format(len(all_files), max_filesize))

    # remove wav files bigger than max_filesize and get the biggest file path
    biggest_file_path = ""
    biggest_file_size = 0
    files = []
    for wav in all_files:
        wav_path = os.path.join(path, wav)
        if os.path.getsize(wav_path) < max_filesize:
            files.append(wav)
            if os.path.getsize(wav_path) > biggest_file_size:
                biggest_file_path = wav_path
                biggest_file_size = os.path.getsize(wav_path)

    print("loaded {} files after capping max_filesize of {}".format(len(files), max_filesize))

    # get the biggest file's frame count and set it as the max pad length
    wave, sr = librosa.load(biggest_file_path, mono=True)
    biggest_mfcc = librosa.feature.mfcc(wave, sr)
    print("The biggest file's frame count is ", biggest_mfcc.shape[1])
    pad_width = num_features
    print("The number of frame is set to ", pad_width)
    while True:
        for wav in files:

            if not wav.endswith(".wav"): continue

            wav_path = os.path.join(path, wav)
            wave, sr = librosa.load(wav_path, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width - len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                # print(np.array(batch_features).shape)
                # yield np.array(batch_features), labels
                yield batch_features  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch

