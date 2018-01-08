# helper codes borrowed from https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow.

from functools import partial
from contextlib import closing
from multiprocessing import Pool
from tqdm import tqdm
from pydub import AudioSegment
from pydub import silence
import os

def read_audio(audio_path):
    return AudioSegment.from_file(audio_path)


def remove_file(path):
    if os.path.exists(path):
        print(" [*] Removed: {}".format(path))
        os.remove(path)

def split_on_silence_with_pydub(
        audio_path, skip_idx=0, out_ext="wav",
        silence_thresh=-40, min_silence_len=400,
        silence_chunk_len=100, keep_silence=100):

    filename = os.path.basename(audio_path).split('.', 1)[0]
    in_ext = audio_path.rsplit(".")[1]

    audio = read_audio(audio_path)
    not_silence_ranges = silence.detect_nonsilent(
        audio, min_silence_len=silence_chunk_len,
        silence_thresh=silence_thresh)

    edges = [not_silence_ranges[0]]

    for idx in range(1, len(not_silence_ranges)-1):
        cur_start = not_silence_ranges[idx][0]
        prev_end = edges[-1][1]

        if cur_start - prev_end < min_silence_len:
            edges[-1][1] = not_silence_ranges[idx][1]
        else:
            edges.append(not_silence_ranges[idx])

    audio_paths = []
    for idx, (start_idx, end_idx) in enumerate(edges[skip_idx:]):
        start_idx = max(0, start_idx - keep_silence)
        end_idx += keep_silence

        target_audio_path = "{}/{}-{:04d}.{}".format(
                os.path.dirname(audio_path), filename, idx, out_ext)

        audio[start_idx:end_idx].export(target_audio_path, out_ext)

        audio_paths.append(target_audio_path)


    return audio_paths


def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        # closing is a context manager that closes tool automatically
        with closing(Pool()) as pool:
            for out in tqdm(pool.imap_unordered(fn, items), 
                            total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, 
                         total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results

def split_on_silence_batch(audio_paths, parallel = True , **kargv):

    audio_paths.sort()

    fn = partial(split_on_silence_with_pydub, **kargv)
        
    parallel_run(fn, audio_paths,
            desc="Split on silence", parallel=parallel)

