# +
import os
import wget
import json
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from nemo.collections.asr.models import ClusteringDiarizer
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_meta_inform():
    data_dir = os.path.join('data')
    os.makedirs(data_dir, exist_ok=True)
    meta = {
        'audio_filepath': '', 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': 2, 
        'rttm_filepath': None, 
        'uem_filepath' : None
    }
    with open('data/input_manifest.json','w') as fp:
        json.dump(meta,fp)
        fp.write('\n')

    output_dir = os.path.join('oracle_vad')
    os.makedirs(output_dir,exist_ok=True)
    
    return data_dir
    
def get_model(data_dir):
    MODEL_CONFIG = os.path.join(data_dir,'offline_diarization.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization.yaml"
        MODEL_CONFIG = wget.download(config_url,data_dir)

    config = OmegaConf.load(MODEL_CONFIG)
#     print(OmegaConf.to_yaml(config))
    pretrained_vad = 'vad_marblenet'
    pretrained_speaker_model = 'titanet_large'
    
    output_dir = os.path.join('outputs')
    #config.diarizer.manifest_filepath = 'data/input_manifest.json'
    config.diarizer.out_dir = output_dir #Directory to store intermediate files and prediction outputs

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = 0.50
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.250
    config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
    config.diarizer.clustering.parameters.oracle_num_speakers=False
    config.diarizer.clustering.parameters.num_speakers = 3

    #Here we use our inhouse pretrained NeMo VAD 
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.window_length_in_sec = 0.15
    config.diarizer.vad.shift_length_in_sec = 0.01
    config.diarizer.vad.parameters.onset = 0.8 
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.min_duration_on = 0.1
    config.diarizer.vad.parameters.min_duration_off = 0.4
    
    sd_model = ClusteringDiarizer(cfg=config)
    
    return sd_model
