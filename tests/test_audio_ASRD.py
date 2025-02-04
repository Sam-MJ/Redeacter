import audio_ASR_Diarization
from pathlib import Path

test_file = Path("tests/test_data/an4_diarize_test.wav")

def test_ASRD():
    audio_ASR_Diarization.ASRDiarize(test_file)
