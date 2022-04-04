# TextToSpeech Yury Dud

## Requirements
create container with CUDA cards
```
git pull nvcr.io/nvidia/nemo:1.5.1 or above
```
```
git clone https://github.com/2Bye/TTS_YDud.git
cd TTS_YDud
pip install -r req.txt
```

download ASR checkpoint https://github.com/sberdevices/golos

## Data preprocessing

To create a dataset, run the laptops in this sequence

* 1_download_and_convert.ipynb
* 2_Speaker_diarization.ipynb
* 3_Clastering_speaker_embedding.ipynb (remove emissions/bad data)
* 4_Denoise_and_ASR_predicts.ipynb

## TTS model
Architecture was chosen
```
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch
```
you need to download dataset and and train the model on it
```
https://ruslan-corpus.github.io
```
Next, you take the resulting dataset and finetune the model, which was obtained as a result of Ruslan's training

Download **YD_TTS** checkpoint and **Vocoder** checkpoint
```
https://catalog.ngc.nvidia.com/orgs/nvidia/models/waveglow_ckpt_amp_256
https://drive.google.com/file/d/1pmR7NeTBAVOndeypH9OnlJemCGSm9EdT/view?usp=sharing
```

Change the paths to the checkpoints in the file **server.py**

Run **server.py** in **websocket_service** dir

Run **client.py**
