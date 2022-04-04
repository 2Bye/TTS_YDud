# -*- coding: utf-8 -*-
import argparse
import models
import time
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import asyncio
import websockets
import json
import io
import base64

import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from torch.nn.utils.rnn import pad_sequence

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common import utils
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import TextProcessing
from pitch_transform import pitch_transform_custom
from waveglow import model as glow
from waveglow.denoiser import Denoiser
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.modules['glow'] = glow

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--save-mels', action='store_true', help='')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str,
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--waveglow', type=str,
                        help='Full path to the WaveGlow model checkpoint file (skip to only generate mels)')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Warmup iterations before measuring performance')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')
    parser.add_argument('--speaker', type=int, default=0,
                        help='Speaker ID for a multi-speaker model')

    parser.add_argument('--p-arpabet', type=float, default=0.0, help='')
    parser.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                        help='')
    parser.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                        help='')
    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=10,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=0.8,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['basic_cleaners'], type=str,
                                 help='Type of text cleaners for input text')
#     text_processing.add_argument('--symbol-set', type=str, default='english_basic_withoit_tokens',
#                                  help='Define symbol set for input text')

#     text_processing.add_argument('--symbol-set', type=str, default='english_basic',
#                                  help='Define symbol set for input text')

    text_processing.add_argument('--symbol-set', type=str, default='russian_basic',
                                 help='Define symbol set for input text')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=2,
                      help='Number of speakers in the model.')
    return parser

def load_model_from_ckpt(checkpoint_path, ema, model):

    checkpoint_data = torch.load(checkpoint_path)
    #del checkpoint_data['state_dict']['module.encoder.word_emb.weight']
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')

    return model

def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)
    print(model_config)
    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, ema, model)

    if model_name == "WaveGlow":
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

        model = model.remove_weightnorm(model)

    if amp:
        model.half()
    model.eval()
    return model.to(device)

def load_fields(text):
    return {'text' : [text]}

def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False,
                           load_pitch=False, p_arpabet=0.0):
    
    tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)
    
    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])
    print(fields['text'])
    for t in fields['text']:
        print(tp.sequence_to_text(t.numpy()))

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches

def build_pitch_transformation(args):
    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                    - mean) / std
        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')

class MeasureTime(list):
    def __init__(self, *args, cuda=True, **kwargs):
        super(MeasureTime, self).__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime((sum(ab) for ab in zip(self, other)), cuda=cuda)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference', allow_abbrev=False)
parser = parse_args(parser)
args, unk_args = parser.parse_known_args()
fastpitch = 'checkpoints/'
waveglow = 'checkpoints/'
device='cuda'

generator = load_and_setup_model(
            'FastPitch', parser, fastpitch, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, ema=args.ema,
            jitable=args.torchscript)

gen_measures = MeasureTime(cuda=args.cuda)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    waveglow = load_and_setup_model(
                'WaveGlow', parser, waveglow, args.amp, device,
                unk_args=unk_args, forward_is_infer=True, ema=args.ema)
    denoiser = Denoiser(waveglow).to(device)
    waveglow = getattr(waveglow, 'infer', waveglow)
    waveglow_measures = MeasureTime(cuda=args.cuda)

gen_kw = {'pace': args.pace,
          'speaker': args.speaker,
          'pitch_tgt': None,
          'pitch_transform': build_pitch_transformation(args)}

def tts(text, generator_fp, generator_hifi = None, waveglow = None, vocoder=None):
    
    fields = load_fields(text)
    
    batches = prepare_input_sequence(
        fields, device, args.symbol_set, args.text_cleaners, 1,
        args.dataset_path, load_mels=(generator is None), p_arpabet=args.p_arpabet)

    for b in batches:
        with torch.no_grad(), gen_measures:
            mel, mel_lens, *_ = generator(b['text'], **gen_kw)
    print(mel.shape)
            
            
    if vocoder == 'waveglow':
        total_audio = np.array([]).astype('int16')
        with torch.no_grad(), waveglow_measures:
            audios = waveglow(mel, sigma=args.sigma_infer)
            audios = denoiser(audios.float(), strength=args.denoising_strength).squeeze(1)
            for i, audio in enumerate(audios):
                audio = audio[:mel_lens[i].item() * args.stft_hop_length]
                audio = audio / torch.max(torch.abs(audio))
                audio = audio.cpu().numpy()
                total_audio = np.append(total_audio, audio)
    else:
        with torch.no_grad():
                #mel = torch.FloatTensor(mel.cpu()).to(device)
                y_g_hat = generator_hifi(mel)
                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')
            
    return audio

async def ws(websocket, path):
    async for data in websocket:
        data = json.loads(data)
        try:
            text = data['text']
            au = tts(text, generator, waveglow=waveglow, vocoder='waveglow')
            wav_file = io.BytesIO()
            write(wav_file, 22050, au)
            enc = base64.b64encode(wav_file.read())
            enc = str(enc)[2:-1]
            data['event'] = 'success'
            data['audio'] = enc
            await websocket.send(json.dumps(data))

        except Exception as e:
            print("Error: {}".format(e))
            await websocket.send(json.dumps({'event': 'error', 'msg': "Ошибка запуска! {}".format(e)}))

print('WS Server started.\n')
asyncio.get_event_loop().run_until_complete(websockets.serve(ws, '0.0.0.0', '1234', max_size=1024*1024*10))
asyncio.get_event_loop().run_forever()

