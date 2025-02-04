from pathlib import Path
import audio_ASR_Diarization
import audio_edit

input_file = Path("tests/test_data/an4_diarize_test.wav")
SPEAKER_ID = 0

audio_ASR_Diarization.ASRDiarize(input_file)

rttm = audio_edit.read_rttm("ASRD_output/pred_rttms/an4_diarize_test.rttm")
speaker_data = audio_edit.rttm_to_speaker_data(rttm)
individual_speaker_timecode = speaker_data[SPEAKER_ID]

audio_data = audio_edit.read_audio(input_file)
individual_speaker_data = audio_edit.get_speaker_data(audio_data, individual_speaker_timecode)
audio_buffer = audio_edit.generate_silence(audio_data)
audio_edit.write_speaker_data(audio_buffer, individual_speaker_data, individual_speaker_timecode, audio_data)
audio_edit.write_fades(audio_buffer, individual_speaker_timecode, audio_data)
out_path = audio_edit.construct_out_path(input_file, SPEAKER_ID)
audio_edit.write(audio_buffer, audio_data, out_path)
