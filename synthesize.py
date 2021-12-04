# -*- coding: utf-8 -*-

import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from model.network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
import utils

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('./checkpoints/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    print(text)
    text = t.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = t.zeros([1,1, 80]).cuda()
    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m=m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1)

        print(postnet_pred.shape)
        #mag_pred = m_post.forward(postnet_pred)
        
        
    #wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    #postnet_pred = t.transpose(postnet_pred, 0, 1)
    postnet_pred = t.squeeze(postnet_pred)
    postnet_pred = t.transpose(postnet_pred, 0, 1)
    vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    wav = utils.vocgan_infer(postnet_pred, vocoder)   
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=98000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=62000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=400)

    args = parser.parse_args()
    synthesis("가까운 시",args)
