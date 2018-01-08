import argparse
import download_youtube
import split_on_silence
import batch_helper as bh
from glob import glob
import os
import tflearn
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', type=str, default=os.getcwd())
    parser.add_argument('--moon_url', type=str, default= "moon.txt")
    parser.add_argument('--son_url', type=str, default= "son.txt")
    config = parser.parse_args()

    base_dir = config.base_path
    moon = "Moon"
    son = "Son"
    moon_path = os.path.join(config.base_path, config.moon_url)
    son_path = os.path.join(config.base_path, config.son_url)

    # get youtube for moon
    with open(moon_path,"r") as f:
        moon_list = [line[:-1] for line in f]

    for line in moon_list:
        download_youtube.get_youtube_audio(line, base_dir = base_dir, speaker = moon)

    # get youtube for son
    son_list = []
    with open(son_path, "r") as f:
        for line in f:
            if line[-1] == "\n":
                son_list.append(line[:-1])
            elif len(line) > 2:
                son_list.append(line)
    son_list2 = []
    for i, url in enumerate(son_list):
        line = "{}{:03d}{}{}{}".format("assets/",i+1,".txt|", url, "|0:09")
        son_list2.append(line)

    for line in son_list2:
        download_youtube.get_youtube_audio(line, base_dir = base_dir, speaker = son)


    # grab all the audio files basedon their names
    audio_pattern = os.path.join(base_dir,"audio/*.wav")
    audio_paths = glob(audio_pattern)

    split_on_silence.split_on_silence_batch(audio_paths, parallel = False, out_ext="wav")

    # remove unnecessary audio files
    split_audio_pattern = os.path.join(base_dir,"audio/*-*-*.wav")
    split_audio_paths = glob(split_audio_pattern)

    whole_audio_pattern = os.path.join(base_dir,"audio/*-*.wav")
    whole_audio_paths = glob(whole_audio_pattern)

    for path  in list(set(whole_audio_paths)-set(split_audio_paths)):
        split_on_silence.remove_file(path)

    # count how many are in each class
    moon_pattern = os.path.join(base_dir,"audio/Moon*.wav")
    moon_paths = glob(moon_pattern)
    print ("moon data: ",len(moon_paths))

    Son_pattern = os.path.join(base_dir,"audio/Son*.wav")
    Son_paths = glob(Son_pattern)
    print ("son data:",len(Son_paths))


    # make models


    # Classification
    tflearn.init_graph()

    number_classes=len(bh.get_speakers(os.path.join(base_dir,"audio")))

    net = tflearn.input_data([None, 123,20])
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, number_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    batch = bh.mfcc_batch_generator(batch_size=3000, path = os.path.join(base_dir,"audio"), max_filesize = 500013)

    X,Y=next(batch)
    X = np.swapaxes(X,1,2)    # in LSTM, the timestep has to be the second dimension like [None,123,20]
    Y = np.array(Y)

    trainX, trainY = X[0:2000], Y[0:2000]
    testX, testY = X[2000:], Y[2000:]


    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=32)




    # Save a model
    bh.makedirs(os.path.join(base_dir, "models"))
    model.save(os.path.join(base_dir, 'models/model1.tflearn'))

    # # Load a model
    # model.load(os.path.join(base_dir,'model1.tflearn'))