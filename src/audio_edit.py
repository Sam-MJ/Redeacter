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

def read(sound_file: Path) -> AudioData:
    """Extract audio data from sound file"""
    with soundfile.SoundFile(sound_file, "r") as s:
        data = s.read()
        samplerate=s.samplerate
        subtype=s.subtype
        channels=data.shape[1:]
        dtype=data.dtype

        audio_data = AudioData(path=sound_file, data=data, samplerate=samplerate, subtype=subtype, channels=channels, dtype=dtype)

    return audio_data

def NeMo_timecode_to_speaker_data(timecode: list[str]) -> dict[SpeakerData]:
    """convert nemo list of strings to dictionary of speaker data."""
    speakers = {}

    for speaker in timecode:
        s = speaker.split(" ")
        start = float(s[0])
        end = float(s[1])
        speaker_number = int(s[2].split("_")[1])

        if speaker_number not in speakers:
            speaker_data = SpeakerData(speaker_number, [start], [end])
            speakers[speaker_number] = speaker_data
        else:
            speakers[speaker_number].start.extend(start)
            speakers[speaker_number].end.extend(end)

    return speakers

def get_speaker_data(audio_data: AudioData, speaker_data: SpeakerData) -> list[ndarray]:
    """returns a list of numpy views of a speaker data from array for a specific speaker""" # should this be a copy?
    assert len(speaker_data.starts) == len(speaker_data.ends)

    all_data_for_given_speaker = []
    for i in range(len(speaker_data.starts)):
        start = speaker_data.starts[i]
        end = speaker_data.ends[i]

        all_data_for_given_speaker.append(audio_data.data[int(audio_data.samplerate * float(start)): int(audio_data.samplerate * float(end))])

    return all_data_for_given_speaker


def generate_silence(audio_data: AudioData) -> ndarray:
    # create silence block
    silence_samples_number = audio_data.data.shape[0]
    shape = (int(silence_samples_number),) + audio_data.channels
    silence_block = numpy.zeros((shape), audio_data.dtype)

    return silence_block

def write_speaker_data(silence_block, all_data_for_given_speaker: list[ndarray], speaker_data: SpeakerData, audio_data: AudioData) -> numpy.ndarray:
    """write each of the speaker data blocks onto the block of silence"""
    assert len(speaker_data.starts) == len(all_data_for_given_speaker)

    for i in range(len(speaker_data.starts)):
        start = speaker_data.starts[i]
        start_sample = int(start * audio_data.samplerate)
        speaker_data = all_data_for_given_speaker[i]

    block_with_speaker_data = numpy.insert(silence_block, start_sample, speaker_data)
    return block_with_speaker_data

def write_fades(block_with_speaker_data: ndarray, speaker_data: SpeakerData, audio_data: AudioData):
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

    for i in range(len(speaker_data.starts)):
        # fade in
        start_sample = int(speaker_data.starts[i] * audio_data.samplerate)
        fade_in(block_with_speaker_data, start_sample, fade_samples, fade_curve)
        # fade out
        end_sample = int(speaker_data.ends[i] * audio_data.samplerate)
        fade_out(block_with_speaker_data, end_sample, fade_samples, fade_curve)


def construct_out_path(in_path: Path, speaker_number):
    new_stem = in_path.stem + f"_speaker_{speaker_number}"
    suffix = in_path.suffix
    outpath = in_path.parent.joinpath(new_stem + suffix)
    return outpath

def write(block_with_speaker_data: ndarray, audio_data: AudioData, out_path):
    """write numpy array to .wav file at output path"""
    soundfile.write(out_path, block_with_speaker_data, audio_data.samplerate, audio_data.subtype)
