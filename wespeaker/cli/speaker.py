# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#                    Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
import kaldiio
from tqdm import tqdm

from wespeaker.cli.hub import Hub
from wespeaker.cli.utils import get_args
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.diar.umap_clusterer import cluster
from wespeaker.diar.extract_emb import subsegment
from wespeaker.diar.make_rttm import merge_segments
from wespeaker.utils.utils import set_seed


class Speaker:

    def __init__(self, model_dir: str):
        set_seed()
        config_path = os.path.join(model_dir, 'config.yaml')
        model_path = os.path.join(model_dir, 'avg_model.pt')
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.model = get_speaker_model(configs['model'])(**configs['model_args'])
        load_checkpoint(self.model, model_path)
        self.model.eval()
        self.vad = load_silero_vad()
        self.table = {}
        self.resample_rate = 16000
        self.apply_vad = False
        self.device = torch.device('cpu')
        self.wavform_norm = False

        self.diar_min_duration = 0.255
        self.diar_window_secs = 1.5
        self.diar_period_secs = 0.75
        self.diar_frame_shift = 10
        self.diar_batch_size = 32
        self.diar_subseg_cmn = True

    def set_wavform_norm(self, wavform_norm: bool):
        self.wavform_norm = wavform_norm

    def set_resample_rate(self, resample_rate: int):
        self.resample_rate = resample_rate

    def set_vad(self, apply_vad: bool):
        self.apply_vad = apply_vad

    def set_device(self, device: str):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

    def set_diarization_params(self, min_duration=0.255, window_secs=1.5,
                               period_secs=0.75, frame_shift=10,
                               batch_size=32, subseg_cmn=True):
        self.diar_min_duration = min_duration
        self.diar_window_secs = window_secs
        self.diar_period_secs = period_secs
        self.diar_frame_shift = frame_shift
        self.diar_batch_size = batch_size
        self.diar_subseg_cmn = subseg_cmn

    def compute_fbank(self, wavform, sample_rate=16000,
                      num_mel_bins=80, frame_length=25,
                      frame_shift=10, cmn=True):
        feat = kaldi.fbank(wavform,
                           num_mel_bins=num_mel_bins,
                           frame_length=frame_length,
                           frame_shift=frame_shift,
                           sample_frequency=sample_rate,
                           window_type='hamming')
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def extract_embedding_feats(self, fbanks, batch_size, subseg_cmn):
        fbanks_array = np.stack(fbanks)
        if subseg_cmn:
            fbanks_array = fbanks_array - np.mean(fbanks_array, axis=1, keepdims=True)
        embeddings = []
        fbanks_array = torch.from_numpy(fbanks_array).to(self.device)
        for i in tqdm(range(0, fbanks_array.shape[0], batch_size)):
            batch_feats = fbanks_array[i:i + batch_size]
            with torch.no_grad():
                batch_embs = self.model(batch_feats)
                batch_embs = batch_embs[-1] if isinstance(batch_embs, tuple) else batch_embs
            embeddings.append(batch_embs.detach().cpu().numpy())
        return np.vstack(embeddings)

    def extract_embedding(self, audio_path: str):
        pcm, sample_rate = torchaudio.load(audio_path, normalize=self.wavform_norm)
        return self.extract_embedding_from_pcm(pcm, sample_rate)

    def extract_embedding_from_pcm(self, pcm: torch.Tensor, sample_rate: int):
        if self.apply_vad:
            wav = pcm.mean(dim=0, keepdim=True) if pcm.size(0) > 1 else pcm
            if sample_rate != 16000:
                wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav)
            segments = get_speech_timestamps(wav, self.vad, return_seconds=True)
            pcmTotal = torch.Tensor()
            for seg in segments:
                start = int(seg['start'] * sample_rate)
                end = int(seg['end'] * sample_rate)
                pcmTemp = pcm[0, start:end]
                pcmTotal = torch.cat([pcmTotal, pcmTemp], 0)
            if pcmTotal.numel() == 0:
                return None
            pcm = pcmTotal.unsqueeze(0)
        if sample_rate != self.resample_rate:
            pcm = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)(pcm)
        feats = self.compute_fbank(pcm, sample_rate=self.resample_rate, cmn=True)
        feats = feats.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        return outputs[0].cpu()

    def extract_embedding_list_from_paths(self, audio_paths: list[str]):
        names = []
        embeddings = []
        for audio_path in tqdm(audio_paths):
            name = os.path.basename(audio_path).rsplit('.', 1)[0]
            embedding = self.extract_embedding(audio_path)
            if embedding is not None:
                names.append(name)
                embeddings.append(embedding.detach().numpy())
        return names, embeddings


def load_model(language: str) -> Speaker:
    model_path = Hub.get_model(language)
    return Speaker(model_path)


def load_model_local(model_dir: str) -> Speaker:
    return Speaker(model_dir)


def main():
    args = get_args()
    if args.pretrain == "":
        if args.campplus:
            model = load_model("campplus")
            model.set_wavform_norm(True)
        elif args.eres2net:
            model = load_model("eres2net")
            model.set_wavform_norm(True)
        elif args.vblinkp:
            model = load_model("vblinkp")
        elif args.vblinkf:
            model = load_model("vblinkf")
        else:
            model = load_model(args.language)
    else:
        model = load_model_local(args.pretrain)

    model.set_resample_rate(args.resample_rate)
    model.set_vad(args.vad)
    model.set_device(args.device)
    model.set_diarization_params(min_duration=args.diar_min_duration,
                                 window_secs=args.diar_window_secs,
                                 period_secs=args.diar_period_secs,
                                 frame_shift=args.diar_frame_shift,
                                 batch_size=args.diar_emb_bs,
                                 subseg_cmn=args.diar_subseg_cmn)

    if args.task == 'embedding':
        os.makedirs(args.output_file, exist_ok=True)
        names, embeddings = model.extract_embedding_list_from_paths(args.audio_file)
        for name, emb in zip(names, embeddings):
            np.savetxt(os.path.join(args.output_file, f"{name}.txt"), emb)
        print(f"Saved {len(names)} embeddings to {args.output_file}")

    else:
        print(f'Unsupported task {args.task}')
        sys.exit(-1)


if __name__ == '__main__':
    main()
