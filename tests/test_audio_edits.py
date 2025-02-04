import audio_edit
import numpy
from pathlib import Path


test_file = Path("tests/test_data/an4_diarize_test.wav")
test_timecode = ['0.07 2.695 speaker_0', '2.695 5.199999999999999 speaker_1']
test_path = Path("/this/is/a/test.wav")

def test_read():
    audio_data = audio_edit.read(test_file)
    assert audio_data.path == Path('tests/test_data/an4_diarize_test.wav')
    assert audio_data.data.shape == (83200,)
    assert audio_data.samplerate == 16000
    assert audio_data.subtype == "PCM_16"
    assert audio_data.channels == ()
    assert audio_data.dtype == "float64"

def test_NeMo_timecode_to_speaker_data():

    speakers = audio_edit.NeMo_timecode_to_speaker_data(test_timecode)

    assert type(speakers[0].speaker_number) == int
    assert type(speakers[0].starts) == list
    assert type(speakers[0].starts[0]) == float

    assert speakers[0].starts == [0.07]
    assert speakers[1].starts == [2.695]
    assert speakers[0].ends == [2.695]
    assert speakers[1].ends == [5.199999999999999]

def test_get_speaker_data():
    audio_data = audio_edit.read(test_file)
    speakers = audio_edit.NeMo_timecode_to_speaker_data(test_timecode)
    speaker0_data = speakers[0]
    given_speaker_data = audio_edit.get_speaker_data(audio_data, speaker0_data)

    duration = speaker0_data.ends[0] - speaker0_data.starts[0]
    assert given_speaker_data[0].shape[0] / audio_data.samplerate == duration


def test_generate_silence():
    audio_data = audio_edit.read(test_file)
    silence_block = audio_edit.generate_silence(audio_data)

    # assert all values in array are 0
    assert not silence_block.any()

def test_write_speaker_data():
    audio_data = audio_edit.read(test_file)
    silence_block = audio_edit.generate_silence(audio_data)
    speakers = audio_edit.NeMo_timecode_to_speaker_data(test_timecode)
    speaker0_data = speakers[0]
    all_data_for_given_speaker = audio_edit.get_speaker_data(audio_data, speaker0_data)

    # assert array has some non-zero values
    block_with_speaker_data = audio_edit.write_speaker_data(silence_block, all_data_for_given_speaker, speaker0_data, audio_data)
    assert block_with_speaker_data.any()

    # assert the sample at the start is the same as the first sample of that speaker
    start_sample = int(speaker0_data.starts[0] * audio_data.samplerate)
    assert block_with_speaker_data[start_sample] == all_data_for_given_speaker[0][0]

    # what happens if the speaker data is right at the end, is a timestamp accurate enough to fit the right samples?

def test_write_fades():
    audio_data = audio_edit.read(test_file)
    silence_block = audio_edit.generate_silence(audio_data)
    speakers = audio_edit.NeMo_timecode_to_speaker_data(test_timecode)
    speaker0_data = speakers[0]
    all_data_for_given_speaker = audio_edit.get_speaker_data(audio_data, speaker0_data)
    block_with_speaker_data = audio_edit.write_speaker_data(silence_block, all_data_for_given_speaker, speaker0_data, audio_data)

    previous = block_with_speaker_data.copy()
    audio_edit.write_fades(block_with_speaker_data, speaker0_data, audio_data)

    assert not numpy.array_equal(previous, block_with_speaker_data)
    # I'm not sure how to test actual fades, but it sounds alright?


def test_construct_output_path():
    assert audio_edit.construct_out_path(test_path, 3) == Path("/this/is/a/test_speaker_3.wav")

def test_write():
    audio_data = audio_edit.read(test_file)
    silence_block = audio_edit.generate_silence(audio_data)
    speakers = audio_edit.NeMo_timecode_to_speaker_data(test_timecode)
    speaker0_data = speakers[0]
    all_data_for_given_speaker = audio_edit.get_speaker_data(audio_data, speaker0_data)
    block_with_speaker_data = audio_edit.write_speaker_data(silence_block, all_data_for_given_speaker, speaker0_data, audio_data)

    audio_edit.write_fades(block_with_speaker_data, speaker0_data, audio_data)
    output_path = audio_edit.construct_out_path(test_file, 0)
    audio_edit.write(block_with_speaker_data, audio_data, output_path)
