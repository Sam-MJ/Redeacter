import soundfile
from dataclasses import dataclass
import numpy
from numpy import ndarray
from pathlib import Path


@dataclass
class AudioData():
    path: Path
    data: ndarray
    samplerate: int
    subtype: int
    channels: int
    dtype: int

@dataclass
class SpeakerData():
    speaker_number: int
    starts: list[float]
    ends: list[float]

def read_audio(sound_file: Path) -> AudioData:
    """Extract audio data from sound file"""
    with soundfile.SoundFile(sound_file, "r") as s:
        data = s.read()
        samplerate=s.samplerate
        subtype=s.subtype
        channels=data.shape[1:]
        dtype=data.dtype

        audio_data = AudioData(path=sound_file, data=data, samplerate=samplerate, subtype=subtype, channels=channels, dtype=dtype)

    return audio_data

def read_rttm(rttm_file: Path):
    "SPEAKER an4_diarize_test 1   0.220   2.875 <NA> <NA> speaker_1 <NA> <NA>"

    with open(rttm_file, "r", encoding="utf-8") as f:
       lines = f.readlines()

    return lines

def rttm_to_speaker_data(rttm_output: list[str]) -> dict[SpeakerData]:
    """convert nemo list of strings to dictionary of speaker data."""
    speakers = {}

    for speaker in rttm_output:
        s = speaker.split(" ")
        start = float(s[5])
        duration = float(s[8])
        end = start + duration
        speaker_number = int(s[11].split("_")[1])

        if speaker_number not in speakers:
            individual_speaker_data = SpeakerData(speaker_number, [start], [end])
            speakers[speaker_number] = individual_speaker_data
        else:
            speakers[speaker_number].start.extend(start)
            speakers[speaker_number].end.extend(end)

    return speakers

def get_speaker_data(audio_data: AudioData, individual_speaker_data: SpeakerData) -> list[ndarray]:
    """returns a list of numpy views of a speaker data from array for a specific speaker""" # should this be a copy?
    assert len(individual_speaker_data.starts) == len(individual_speaker_data.ends)

    all_data_for_given_speaker = []
    for i in range(len(individual_speaker_data.starts)):
        start = individual_speaker_data.starts[i]
        end = individual_speaker_data.ends[i]

        all_data_for_given_speaker.append(audio_data.data[int(audio_data.samplerate * float(start)): int(audio_data.samplerate * float(end))])

    return all_data_for_given_speaker


def generate_silence(audio_data: AudioData) -> ndarray:
    # create silence block
    silence_samples_number = audio_data.data.shape[0]
    shape = (int(silence_samples_number),) + audio_data.channels
    silence_block = numpy.zeros((shape), audio_data.dtype)

    return silence_block

def write_speaker_data(silence_block: ndarray, all_data_for_given_speaker: list[ndarray], individual_speaker_data: SpeakerData, audio_data: AudioData) -> numpy.ndarray:
    """write each of the speaker data blocks onto the block of silence"""
    # assert list of timestamps is the same as the list of data blocks
    assert len(individual_speaker_data.starts) == len(all_data_for_given_speaker)

    def ms_to_sample(time_in_ms, samplerate):
        return int(time_in_ms * samplerate)


    for i in range(len(individual_speaker_data.starts)):
        start = individual_speaker_data.starts[i]
        start_sample = ms_to_sample(start, audio_data.samplerate)

        end = individual_speaker_data.ends[i]
        end_sample = ms_to_sample(end, audio_data.samplerate)

        # assert samples are the right way round and that it doesn't go off the end of the file.
        assert start_sample < end_sample, "start sample is after the end sample"
        assert end_sample <= silence_block.shape[0], "end sample is placed outside of the end of the array"

        individual_speaker_data = all_data_for_given_speaker[i]
        silence_block[start_sample:end_sample] = individual_speaker_data

def write_fades(block_with_speaker_data: ndarray, individual_speaker_data: SpeakerData, audio_data: AudioData):
    """Write short fade in and fade outs on every cut."""

    def fade_in(audio: ndarray, start: int, fade_length: int, fade_curve: ndarray):
        # apply the curve, how do you do fade in and out? is it the same?
        audio[start:start + fade_length] = audio[start:start + fade_length] * fade_curve

    def fade_out(audio: ndarray, start: int, fade_length: int, fade_curve: ndarray):
        # apply the curve, how do you do fade in and out? is it the same?
        reversed_curve = fade_curve[::-1]
        audio[start - fade_length:start] = audio[start- fade_length:start] * reversed_curve

    fade_length = 0.2
    fade_samples = int(fade_length * audio_data.samplerate)
    fade_curve = numpy.linspace(0.0, 1.0, fade_samples) # has to be numbers between 1 and 0, can be log/reverse log

    for i in range(len(individual_speaker_data.starts)):
        # fade in
        start_sample = int(individual_speaker_data.starts[i] * audio_data.samplerate)
        fade_in(block_with_speaker_data, start_sample, fade_samples, fade_curve)
        # fade out
        end_sample = int(individual_speaker_data.ends[i] * audio_data.samplerate)
        fade_out(block_with_speaker_data, end_sample, fade_samples, fade_curve)


def construct_out_path(in_path: Path, speaker_number):
    new_stem = in_path.stem + f"_speaker_{speaker_number}"
    suffix = in_path.suffix
    outpath = in_path.parent.joinpath(new_stem + suffix)
    return outpath

def write(block_with_speaker_data: ndarray, audio_data: AudioData, out_path):
    """write numpy array to .wav file at output path"""
    soundfile.write(out_path, block_with_speaker_data, audio_data.samplerate, audio_data.subtype)
