import time
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# 言語処理
# import fasttext
# import fasttext.util
from transformers import BertTokenizer, BertModel

# データ処理
import numpy as np
import torch

def extract_feats(config):
    start = time.time()

    # FPs
    with open(config.fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Prepare bert
    bert_model_dir = Path(config.bert_model_dir)
    vocab_file_path = bert_model_dir / "vocab.txt"
    bert_tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, do_basic_tokenize=False)
    bert_model = BertModel.from_pretrained(bert_model_dir)
    bert_model.eval()
    def preprocess_ipu(speaker_id, koen_id, ipu_id, ipu_tagtext, in_dir, out_dir):

        # get tokens and fp labels
        fp_labels = [0]     # fps sometimes appear at the head of the breath group
        tokens = ["[CLS]"]
        for m in ipu_tagtext.split(" "):
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
        np.save(in_dir / f"{speaker_id}-{koen_id}-{ipu_id}-feats.npy", outputs_numpy)
        np.save(out_dir / f"{speaker_id}-{koen_id}-{ipu_id}-feats.npy", np.array(fp_labels))

    # extraxt features
    infeats_dir = Path(config.out_dir) / "infeats"
    outfeats_dir = Path(config.out_dir) / "outfeats"
    infeats_dir.mkdir(parents=True, exist_ok=True)
    outfeats_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(config.out_dir) / f"ipu.list", "r") as f:
        ipus = [tuple(l.split(":")) for l in f.readlines()]
    with torch.no_grad():
        for speaker_id, koen_id, ipu_id, ipu in tqdm(ipus):
            preprocess_ipu(speaker_id, koen_id, ipu_id, ipu, infeats_dir, outfeats_dir)

    # count time
    n_ipu = len(ipus)
    elapsed_time = time.time() - start
    time_log = "elapsed_time of feature extraction: {} [sec]".format(elapsed_time)
    time_log_ipu = "elapsed_time of feature extraction (per IPU): \
        {} [sec]".format(elapsed_time / n_ipu)
    print(time_log + "\n" + time_log_ipu)
    with open(Path(config.out_dir) / "time.log", "w") as f:
        f.write(time_log + "\n" + time_log_ipu)

def extract_feats_test(config, utt_list_name):
    start = time.time()

    # FPs
    with open(config.fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Prepare bert
    bert_model_dir = Path(config.bert_model_dir)
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
    infeats_dir = Path(config.out_dir) / "infeats"
    outfeats_dir = Path(config.out_dir) / "outfeats"
    infeats_dir.mkdir(parents=True, exist_ok=True)
    outfeats_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(config.out_dir) / "{}.list".format(utt_list_name), "r") as f:
        utts = [tuple(l.split(":")) for l in f.readlines()]
    with torch.no_grad():
        for utt_id, utt in tqdm(utts):
            preprocess_utt(utt_id, utt, infeats_dir, outfeats_dir)

    # count time
    n_utt = len(utts)
    elapsed_time = time.time() - start
    time_log ="elapsed_time of feature extraction: {} [sec]".format(elapsed_time)
    time_log_utt ="elapsed_time of feature extraction (per utt): \
        {} [sec]".format(elapsed_time / n_utt)
    print(time_log + "\n" + time_log_utt)
    with open(Path(config.out_dir) / "time.log", "w") as f:
        f.write(time_log + "\n" + time_log_utt)

@hydra.main(config_path="conf/preprocess", config_name="config")
def main(config: DictConfig):
    extract_feats(config)
    
if __name__=="__main__":
    main()