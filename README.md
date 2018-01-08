# Speaker Identifier
Find out who the speaker is from the youtube videos. 

### Prerequisites

You should install the necessary packages using the requirement.txt 

```
pip install -r /path/to/requirements.txt
```

### Training the model

If you want to download the youtube video and retrain the model yourself, run the following code. 
```
python main.py 
```
If you want to train with different input vidoes, tweak the moon.txt and son.txt files. 

### Running the test

If you want to only run the test using the pretrained model, run the following code. 

```
python test.py
```

For different input videos for the test, tweak the test.txt file. 

### Built With

* python3
* tensorflow
* tflearn

### License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details

### Acknowledgments

* https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow
