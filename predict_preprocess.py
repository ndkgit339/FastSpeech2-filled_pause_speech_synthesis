# General
import random
from pathlib import Path
from tqdm import tqdm

# Config
import hydra
from omegaconf import DictConfig, OmegaConf

# 言語処理
from pyknp import Juman
from transformers import BertTokenizer, BertModel

# データ処理
import numpy as np
import torch

# My library
from preprocessor_predict import extract_feats_test


def process_morph(data_dir):

    juman = Juman()

    with open(Path(data_dir) / f"utt.list", "r") as f:
        utts = [tuple(l.strip().split(":")) for l in f.readlines()]

    out_utts = []
    for utt_id, utt in utts:
        result = juman.analysis(utt)
        utt_morphs = " ".join(
            [m.midasi for m in result.mrph_list()])
        out_utts.append(
            "{}:{}".format(utt_id, utt_morphs))
    
    with open(Path(data_dir) / f"utt_morphs.list", "w") as f:
        f.write("\n".join(out_utts))

def extract_feats_test(fp_list_path, bert_model_dir, data_dir, utt_list_name):

    # FPs
    with open(fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Prepare bert
    bert_model_dir = Path(bert_model_dir)
    vocab_file_path = bert_model_dir / "vocab.txt"
    bert_tokenizer = BertTokenizer(
        vocab_file_path, do_lower_case=False, do_basic_tokenize=False)
    bert_model = BertModel.from_pretrained(bert_model_dir)
    bert_model.eval()
    def preprocess_utt(utt_id, utt, in_dir, out_dir):

        # get tokens and fp labels
        fp_labels = [0]     # fps sometimes appear at the head of the breath group
        tokens = ["[CLS]"]
        for m in utt.split(" "):
            if m.startswith("(F"):
                fp = m.split("(F")[1].split(")")[0]
                if fp in fp_list:
                    fp_labels[-1] = fp_list.index(fp) + 1
            elif m != "":
                tokens.append(m)
                fp_labels.append(0)

        tokens += ["[SEP]"]
        fp_labels.append(0)

        # get embedding
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        token_tensor = torch.Tensor(token_ids).unsqueeze(0).to(torch.long)
        outputs = bert_model(token_tensor)
        outputs_numpy = outputs[0].numpy().squeeze(axis=0).copy()
        
        assert outputs_numpy.shape[0] == np.array(fp_labels).shape[0], \
            "1st array length {} should be equal to 2nd array length {}".format(
                outputs_numpy.shape[0], np.array(fp_labels).shape[0])
        np.save(in_dir / f"{utt_id}-feats.npy", outputs_numpy)
        np.save(out_dir / f"{utt_id}-feats.npy", np.array(fp_labels))

    # extraxt features
    infeats_dir = Path(data_dir) / "infeats"
    outfeats_dir = Path(data_dir) / "outfeats"
    infeats_dir.mkdir(parents=True, exist_ok=True)
    outfeats_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(data_dir) / "{}.list".format(utt_list_name), "r") as f:
        utts = [tuple(l.split(":")) for l in f.readlines()]
    with torch.no_grad():
        for utt_id, utt in tqdm(utts):
            preprocess_utt(utt_id, utt, infeats_dir, outfeats_dir)


@hydra.main(config_path="config_predict", config_name="preprocess")
def main(config: DictConfig):
    # Set random seed
    random.seed(config.random_seed)

    # Save config
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Preprocess
    print("process morphs...")
    process_morph(data_dir)
    print("extract features...")
    extract_feats_test(config.fp_list_path, config.bert_model_dir, data_dir, "utt_morphs")

if __name__ == "__main__":
    main()