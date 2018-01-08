import argparse
import os
import tflearn
import batch_helper as bh
import numpy as np
import download_youtube
import split_on_silence
from glob import glob
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', type=str, default=os.getcwd())
    parser.add_argument('--new_url_file', type=str, default="test.txt")
    parser.add_argument("--model_name", type=str, default="model1.tflearn")
    config = parser.parse_args()

    base_dir = config.base_path

    bh.makedirs(os.path.join(base_dir, "tests"))
    test_dir = os.path.join(base_dir, "tests")

    speakers = bh.get_speakers(os.path.join(base_dir,"audio"))
    new_url_path = os.path.join(config.base_path, config.new_url_file)

    url_list = []
    with open(new_url_path, "r") as f:
        for line in f:
            if line[-1] == "\n":
                url_list.append(line[:-1])
            elif len(line) > 2:
                url_list.append(line)

    url_list2 = []
    for i, url in enumerate(url_list):
        line = "{}{:03d}{}{}".format("assets/", i + 1, ".txt|", url)
        url_list2.append(line)

    for line in url_list2:
        download_youtube.get_youtube_audio(line, base_dir = test_dir, speaker = "test")



    # grab all the audio files basedon their names
    split_audio_pattern = os.path.join(test_dir, "audio/*-*-*.wav")
    split_audio_paths = glob(split_audio_pattern)

    whole_audio_pattern = os.path.join(test_dir, "audio/*-*.wav")
    whole_audio_paths = glob(whole_audio_pattern)

    audio_paths = list(set(whole_audio_paths) - set(split_audio_paths))

    split_on_silence.split_on_silence_batch(audio_paths, parallel=False, out_ext="wav")

    split_audio_pattern = os.path.join(test_dir, "audio/*-*-*.wav")
    split_audio_paths = glob(split_audio_pattern)

    whole_audio_pattern = os.path.join(test_dir, "audio/*-*.wav")
    whole_audio_paths = glob(whole_audio_pattern)

    for path in list(set(whole_audio_paths) - set(split_audio_paths)):
        split_on_silence.remove_file(path)


    # Classification
    tflearn.init_graph()

    number_classes=len(bh.get_speakers(os.path.join(base_dir,"audio")))

    net = tflearn.input_data([None, 123,20])
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, number_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    batch = bh.mfcc_test_X_generator(batch_size=len(split_audio_paths), path=os.path.join(test_dir, "audio"),
                                  max_filesize=500013, num_features=123)

    test_x = next(batch)
    test_x = np.swapaxes(test_x, 1, 2)


    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)

    model.load(os.path.join(base_dir,'models', config.model_name ))

    # test
    result = model.predict(test_x)
    # print (result)
    # print (bh.one_hot_from_item("Son", speakers))
    # print(bh.one_hot_from_item("Moon", speakers))
    # print (speakers)
    for res in result:
        print ("Son:", res[bh.one_hot_from_item("Son", speakers).index(1)],
               "; Moon: ",res[bh.one_hot_from_item("Moon", speakers).index(1)])
