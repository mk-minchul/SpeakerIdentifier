# helper codes borrowed from https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow.

import youtube_dl
from pydub import AudioSegment
import os

def get_youtube_audio(line, base_dir = None, speaker = "NoSpeaker"):
    """line example: assets/001.txt|https://www.youtube.com/watch?v=_YWqWHe8LwE|title|0:56|30:05"""
    
    
    def makedirs(path):
        if not os.path.exists(path):
            print(" [*] Make directories : {}".format(path))
            os.makedirs(path)
            
    def get_mili_sec(time):
        minute, second = time.strip().split(":")
        return (int(minute) * 60 + int(second))*1000
    
    def remove_file(path):
        if os.path.exists(path):
            print(" [*] Removed: {}".format(path))
            os.remove(path)
    
    if base_dir == None: 
        base_dir = os.path.realpath(os.getcwd())
    makedirs(os.path.join(base_dir, "audio"))
    makedirs(os.path.join(base_dir, "assets"))

    if len(line.split("|")) == 2:
        text_path ,video_url = line.split("|")

    elif len(line.split("|")) == 3:
        text_path ,video_url, start_time = line.split("|")
    else: 
        text_path, video_url, title, start_time, end_time = line.split('|')

    original_path = os.path.join(base_dir, "audio", speaker + "-" + os.path.basename(text_path).replace(".txt", ".original.mp3") )
    out_path = os.path.join(base_dir, "audio", speaker + "-" + os.path.basename(text_path).replace(".txt", ".wav"))

    options = {
        'format': 'bestaudio/best',
        'outtmpl': original_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
    }

    if len(line.split("|")) == 2:
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([video_url])
            audio = AudioSegment.from_file(original_path)
            audio.export(out_path, format="wav")

    elif len(line.split("|")) == 3:
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([video_url])
            audio = AudioSegment.from_file(original_path)
            audio[get_mili_sec(start_time):].export(out_path, format="wav")
            
    else: 
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([video_url])
            audio = AudioSegment.from_file(original_path)
            audio[get_mili_sec(start_time):get_mili_sec(end_time)].export(out_path, format="wav")

    remove_file(original_path)