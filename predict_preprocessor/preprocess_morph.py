from pathlib import Path
import hydra
from omegaconf import DictConfig

# 言語処理
from pyknp import Juman


def process_morph(config):

    juman = Juman()

    with open(Path(config.out_dir) / f"utt.list", "r") as f:
        utts = [tuple(l.strip().split(":")) for l in f.readlines()]

    out_utts = []
    for utt_id, utt in utts:
        result = juman.analysis(utt)
        utt_morphs = " ".join(
            [m.midasi for m in result.mrph_list()])
        out_utts.append(
            "{}:{}".format(utt_id, utt_morphs))
    
    with open(Path(config.out_dir) / f"utt_morphs.list", "w") as f:
        f.write("\n".join(out_utts))

@hydra.main(config_path="conf/preprocess", config_name="config")
def myapp(config: DictConfig):
    process_morph(config)

if __name__=="__main__":
    myapp()