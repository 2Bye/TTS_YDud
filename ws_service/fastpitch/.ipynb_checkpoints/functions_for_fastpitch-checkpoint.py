# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import time
import sys 
import json 
import os
import io
import torch
import numpy as np
from scipy.stats import norm
from torch.nn.utils.rnn import pad_sequence
import re

import models
from common import utils
from common.text.text_processing import TextProcessing
from .pitch_transform import pitch_transform_custom

def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True, jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer, jitable=jitable)

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location= device)
        status = ''

        if 'state_dict' in checkpoint_data:
            sd = checkpoint_data['state_dict']
            if ema and 'ema_state_dict' in checkpoint_data:
                sd = checkpoint_data['ema_state_dict']
                status += ' (EMA)'
            elif ema and not 'ema_state_dict' in checkpoint_data:
                print(f'WARNING: EMA weights missing for {model_name}')

            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k,v in sd.items()}
            status += ' ' + str(model.load_state_dict(sd, strict=True))
        else:
            model = checkpoint_data['model']
        print(f'Loaded {model_name}{status}')

    if amp:
        model.half()
    model.eval()
    return model.to(device)

import time
def text_to_phoneme(text,g2p):

    out = ''
    texts = re.findall(r"[\w']+|[.,\"\']", str(text))

    for word in texts:
        if(word != ',' and word != '.'):
            out += ' '.join(g2p(word)) + ' '
        else:
            word = '@ '
            out += word

    columns = ['text']
    out = [out]
    return {c:[o] for c, o in zip(columns, out)}


def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                           batch_size=128, dataset=None, load_mels=False, load_pitch=False):
    tp = TextProcessing(symbol_set, text_cleaners)

    fields['text'] = [torch.LongTensor(tp.encode_text(text))
                      for text in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

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
    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime(sum(ab) for ab in zip(self, other))


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def viseme(text, g2p, pace):
    with open('gruut_ipa/ipa2viseme.json','r') as f:
        pv = json.load(f)
    text = text_to_phoneme(text,g2p)['text']
    text = do_convert('en-us/cmudict', 'ipa', text, ' ')
    text = ''.join(text).replace("ˈ",'').replace(',','').replace('ˌ', '').replace('  ',' ').split(' ')
    print(text)
    visemes_list = []
    for i in text:
        if '@' in i:
            visemes_list.append('sil')
        else:
            visemes_list.append(pv[i])

    durs = torch.load('dur_pred_for_visemes.pt')
    durs = durs[0].tolist()
    #print(durs)

    total = 0
    timestep = [0]
    for i in range(len(durs)):
        total += (durs[i] / (22050 / 256) * 1000) / pace
        #print((durs[i] / (22050 / 256) * 1000) / pace)
        timestep.append(round(total))
    #print(len(timestep), len(visemes_list))
    #print(timestep)

    visemes_durs = dict(zip(timestep, visemes_list))
    os.remove('dur_pred_for_visemes.pt')
    return visemes_durs

def do_convert(src, dest, pronunciation, separator):

    from gruut_ipa import Phoneme, Phonemes
    from gruut_ipa.espeak import espeak_to_ipa, ipa_to_espeak
    from gruut_ipa.sampa import ipa_to_sampa, sampa_to_ipa

    fixed_src_dest = {"ipa", "espeak", "sampa"}
    src_phonemes: typing.Optional[Phonemes] = None
    dest_phonemes: typing.Optional[Phonemes] = None

    if src not in fixed_src_dest:
        src_phonemes = Phonemes.from_language(src)

    if dest not in fixed_src_dest:
        dest_phoneme_map = Phonemes.from_language(dest).gruut_ipa_map

        # ipa -> original phoneme
        dest_phonemes = Phonemes()
        for k, v in dest_phoneme_map.items():
            if v in dest_phonemes.gruut_ipa_map:
                continue

            dest_phonemes.phonemes.append(Phoneme(text=k, is_ipa=False))
            dest_phonemes.ipa_map[v] = k

        dest_phonemes.update()

    if pronunciation:
        # From arguments
        pronunciations = pronunciation
    else:
        # From stdin
        pronunciations = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading pronunciations from stdin...", file=sys.stderr)

    for line in pronunciations:
        line = line.strip()
        if line:
            if src == "ipa":
                src_ipa = line
            elif src == "espeak":
                src_ipa = espeak_to_ipa(line)
            elif src == "sampa":
                src_ipa = sampa_to_ipa(line)
            else:
                assert src_phonemes is not None
                src_ipa = separator.join(
                    src_phonemes.gruut_ipa_map.get(p.text, p.text)
                    for p in src_phonemes.split(line)
                )

            if dest == "ipa":
                dest_pron = src_ipa
            elif dest == "espeak":
                dest_pron = "[[" + ipa_to_espeak(src_ipa) + "]]"
            elif dest == "sampa":
                dest_pron = ipa_to_sampa(src_ipa)
            else:
                assert dest_phonemes is not None
                dest_pron = separator.join(
                    p.text for p in dest_phonemes.split(src_ipa, is_ipa=False)
                )

            sys.stdout.flush()
    return  dest_pron
