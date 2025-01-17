import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import symbols, text_to_sequence
from utils.tools import pad_1D, pad_2D


def convert_lab_available(lab):
    if lab == "sil":
        lab = ""
    elif lab == "A":
        lab = "a"
    elif lab == "I":
        lab = "i"
    elif lab == "U":
        lab = "u"
    elif lab == "E":
        lab = "e"
    elif lab == "O":
        lab = "o"
    elif lab == "cl":
        lab = "q"
    elif lab == "pau":
        lab = "sp"
    elif lab == "v":
        lab = "b"
    return lab

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}

        # Use fp tag
        self.preprocess_use_fp_tag = preprocess_config["preprocessing"]["use_fp_tag"]
        self.train_use_fp_tag = train_config["use_fp_tag"]
        if self.train_use_fp_tag:
            self.basename, self.speaker, self.text, self.raw_text, self.fp_tag = \
                self.process_meta(filename)
        else:
            self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
                filename)

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array([self.symbol_to_id[t] for t in self.text[idx].replace("{", "").replace("}", "").split()])
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            accent = np.array(accent[:len(phone)])

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        if self.use_accent:
            sample["accent"] = accent

        if self.train_use_fp_tag:
            fp_tag = np.array(
                [int(f) for f in self.fp_tag[idx].split(" ")])
            sample["fp_tag"] = fp_tag

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            fp_tag = []
            for line in f.readlines():
                if self.preprocess_use_fp_tag:
                    n, s, t, r, f = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
                    fp_tag.append(f)
                else:
                    n, s, t, r = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
            if self.train_use_fp_tag:
                return name, speaker, text, raw_text, fp_tag
            else:
                return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        if self.use_accent:
            accents = [data[idx]["accent"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        if self.train_use_fp_tag:
            fp_tags = pad_1D([data[idx]["fp_tag"] for idx in idxs])
            fp_tags = np.expand_dims(fp_tags, axis=2)
            if self.use_accent:
                accents = pad_1D(accents)
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                    accents,
                    fp_tags
                )
            else:
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                    fp_tags
                )
        else:
            if self.use_accent:
                accents = pad_1D(accents)
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                    accents
                )
            else:
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations
                )


    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class DatasetForTest(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}

        self.preprocess_with_tag = preprocess_config["preprocessing"]["with_tag"]
        self.with_tag = train_config["with_tag"]
        if self.with_tag:
            self.basename, self.speaker, self.text, self.raw_text, self.filler_tag = self.process_meta(
                filename
            )
        else:
            self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
                filename
            )

        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array([self.symbol_to_id[t] for t in self.text[idx].replace("{", "").replace("}", "").split()])
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            #### Matsunaga ####
            # assert len(phone) == len(accent)
            accent = np.array(accent[:len(phone)])

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        if self.use_accent:
            sample["accent"] = accent

        if self.with_tag:
            filler_tag = np.array([int(f) for f in self.filler_tag[idx].split(" ")])
            sample["f_tag"] = filler_tag

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            filler_tag = []
            for line in f.readlines():
                if self.preprocess_with_tag:
                    n, s, t, r, f = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
                    filler_tag.append(f)
                else:
                    n, s, t, r = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)

            if self.with_tag:
                return name, speaker, text, raw_text, filler_tag
            else:
                return name, speaker, text, raw_text


    def collate_fn(self, data):
        ids = [d["id"] for d in data]
        speakers = [d["speaker"] for d in data]
        texts = [d["text"] for d in data]
        raw_texts = [d["raw_text"] for d in data]
        mels = [d["mel"] for d in data]
        pitches = [d["pitch"] for d in data]
        energies = [d["energy"] for d in data]
        durations = [d["duration"] for d in data]
        if self.use_accent:
            accents = [d["accent"] for d in data]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        if self.with_tag:
            filler_tags = pad_1D([d["f_tag"] for d in data])
            filler_tags = np.expand_dims(filler_tags, axis=2)

        if self.with_tag:
            if self.use_accent:
                accents = pad_1D(accents)
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                    accents,
                    filler_tags
                )
            else:
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                    filler_tags
                )
        else:
            if self.use_accent:
                accents = pad_1D(accents)
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                    accents,
                )
            else:
                return (
                    ids,
                    raw_texts,
                    speakers,
                    texts,
                    text_lens,
                    max(text_lens),
                    mels,
                    mel_lens,
                    max(mel_lens),
                    pitches,
                    energies,
                    durations,
                )


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, train_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.preprocess_use_fp_tag = preprocess_config["preprocessing"]["use_fp_tag"]
        self.train_use_fp_tag = train_config["use_fp_tag"]        
        if self.train_use_fp_tag:
            self.basename, self.speaker, self.text, self.raw_text, self.fp_tag = \
                self.process_meta(filepath)
        else:
            self.basename, self.speaker, self.text, self.raw_text = \
                self.process_meta(filepath)

        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array([self.symbol_to_id[convert_lab_available(t)] for t in self.text[idx].replace("{", "").replace("}", "").split()])
        accent = None
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            accent = np.array(accent[:len(phone)])

        if self.train_use_fp_tag:
            fp_tag = np.array([int(f) for f in self.fp_tag[idx].split(" ")])
            assert len(phone) == len(fp_tag), \
                "phone length {}, ftag length {}, should be equal".format(
                    len(phone), len(fp_tag))
            return (basename, speaker_id, phone, raw_text, accent, fp_tag)
        else:
            return (basename, speaker_id, phone, raw_text, accent)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            fp_tag = []
            for line in f.readlines():
                if self.preprocess_use_fp_tag:
                    n, s, t, r, f = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
                    fp_tag.append(f)
                    assert len(t.split(" ")) == len(f.split(" "))
                else:
                    n, s, t, r = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
            if self.train_use_fp_tag:
                return name, speaker, text, raw_text, fp_tag
            else:
                return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        if self.use_accent:
            accents = pad_1D([d[4] for d in data])

        texts = pad_1D(texts)

        if self.train_use_fp_tag:
            fp_tags = pad_1D([d[5] for d in data])
            fp_tags = np.expand_dims(fp_tags, axis=2)
            if self.use_accent:
                return ids, raw_texts, speakers, texts, text_lens, max(text_lens), accents, fp_tags
            else:
                return ids, raw_texts, speakers, texts, text_lens, max(text_lens), fp_tags
        else:
            if self.use_accent:
                return ids, raw_texts, speakers, texts, text_lens, max(text_lens), accents
            else:
                return ids, raw_texts, speakers, texts, text_lens, max(text_lens), accents


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
