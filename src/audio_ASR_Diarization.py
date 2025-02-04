from pathlib import Path
from omegaconf import OmegaConf
import os
import json

from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_uniqname_from_filepath,
    rttm_to_labels,
    write_rttm2manifest,
)
from nemo.collections.asr.models.msdd_models import NeuralDiarizer


def generate_manifest(audio_filename: Path):
    # Create a manifest file for input with the format described.
    # {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, "label": "infer", "text": "-",
    # "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}

    meta = {
        'audio_filepath': audio_filename.as_posix(),
        'offset': 0,
        'duration': None,
        'label': 'infer',
        'text': '-',
        'num_speakers': None,
        'rttm_filepath': None,
        'uem_filepath': None
    }

    config_dir = Path() / "config_data"
    config_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = config_dir / 'input_manifest.json'
    with manifest_path.open('w') as fp:
        json.dump(meta, fp)
        fp.write('\n')

class NeuralOfflineDiarWithASR(OfflineDiarWithASR):
    """Multiscale diariser, good for audio with overlapping speech, overwriting run_diarization to use Neural Diarizer"""
    # TODO
    # This needs a proper abstraction and tidying up at some point in order to use it properly with ASRD function.
    # Neural diarizer needs a different config I think. it's not just a neat drop in replacement although it does run properly.

    def run_diarization(self, diar_model_config, word_timestamps) -> dict[str, list[str]]:
        """
        Launch the diarization process using the given VAD timestamp (oracle_manifest).

        Args:
            diar_model_config (OmegaConf):
                Hydra configurations for speaker diarization
            word_and_timestamps (list):
                List containing words and word timestamps

        Returns:
            diar_hyp (dict):
                A dictionary containing rttm results which are indexed by a unique ID.
            score Tuple[pyannote object, dict]:
                A tuple containing pyannote metric instance and mapping dictionary between
                speakers in hypotheses and speakers in reference RTTM files.
        """

        if diar_model_config.diarizer.asr.parameters.asr_based_vad:
            self._save_VAD_labels_list(word_timestamps)
            oracle_manifest = os.path.join(self.root_path, 'asr_vad_manifest.json')
            oracle_manifest = write_rttm2manifest(self.VAD_RTTM_MAP, oracle_manifest)
            diar_model_config.diarizer.vad.model_path = None
            diar_model_config.diarizer.vad.external_vad_manifest = oracle_manifest

        diar_model = NeuralDiarizer(cfg=diar_model_config)
        score = diar_model.diarize()
        if diar_model_config.diarizer.vad.model_path is not None and not diar_model_config.diarizer.oracle_vad:
            self._get_frame_level_VAD(
                vad_processing_dir=diar_model.vad_pred_dir,
                smoothing_type=diar_model_config.diarizer.vad.parameters.smoothing,
            )

        diar_hyp = {}
        for k, audio_file_path in enumerate(self.audio_file_list):
            uniq_id = get_uniqname_from_filepath(audio_file_path)
            pred_rttm = os.path.join(self.root_path, 'pred_rttms', uniq_id + '.rttm')
            diar_hyp[uniq_id] = rttm_to_labels(pred_rttm)
        return diar_hyp, score

def ASRDiarize(input_file):
    # Note : The file needs to be mono .wav
    output_path = Path("ASRD_output")
    output_path.mkdir(parents=True, exist_ok=True)

    generate_manifest(input_file)

    MODEL_CONFIG = Path("config_data/diar_infer_telephonic.yaml")

    config = OmegaConf.load(MODEL_CONFIG)

    config.num_workers = os.cpu_count() - 1

    config.diarizer.manifest_filepath = Path("config_data/input_manifest.json").as_posix()
    config.diarizer.out_dir = output_path.as_posix()

    ### generate speech recognition and timestamps
    from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps

    asr_decoder_ts = ASRDecoderTimeStamps(config.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()
    word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

    print("Decoded word output dictionary: \n", word_hyp['an4_diarize_test'])
    print("Word-level timestamps dictionary: \n", word_ts_hyp['an4_diarize_test'])

    # generate diarization, using either OfflineDiarWithASR which performs clusering or NeuralOfflineDiarWithASR for MMSD Neural
    asr_diar_offline = OfflineDiarWithASR(config.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset

    diar_hyp, diar_score = asr_diar_offline.run_diarization(config, word_ts_hyp)
    print("Diarization hypothesis output: \n", diar_hyp['an4_diarize_test'])

    # combine
    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
