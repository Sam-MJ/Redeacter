from pathlib import Path
import audio_ASR_Diarization
import audio_edit

input_file = Path("")

audio_ASR_Diarization.ASRDiarize(input_file)

nemo_timecode =
audio_edit.rttm_to_speaker_data()
