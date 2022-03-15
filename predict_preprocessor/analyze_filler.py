import re
from pathlib import Path

import hydra
from omegaconf import DictConfig

def analyze_fp(config):

    # FPs
    with open(config.fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Get frequency rate of each fp word
    for tagtext_list_path in Path(config.out_dir).glob("*.list"):
        with open(tagtext_list_path, "r") as f:
            tagtext_all = f.read()

        n_position = 0
        for tag_text in tagtext_all.split("\n"):
            if re.fullmatch(r".*?:.*?:.*", tag_text):
                tag_text = tag_text.split(":")[2]
                tag_text = re.sub(r"\(F.*?\)", "", tag_text)
                n_morph = 0
                for t in tag_text.split(" "):
                    if t != "":
                        n_morph += 1
                n_position += n_morph + 1

        n_each_fp_dict = {}
        n_fp = 0
        for fp in fp_list:
            n = tagtext_all.count(f"(F{fp})")
            n_fp += n
            n_each_fp_dict[fp] = n / n_position

        n_fp_all = len(re.findall(r"\(F.*?\)", tagtext_all))
        n_each_fp_dict["others"] = (n_fp_all - n_fp) / n_position
        n_each_fp_dict["no_fp"] = 1 - n_fp_all / n_position


        n_each_fp_text = "\n".join(
            [f"{fp}:{n}" for fp, n in n_each_fp_dict.items()]
        )
        
        with open(tagtext_list_path.parent / (tagtext_list_path.stem + "_fp_rate.list"), "w") as f:
            f.write(n_each_fp_text)

@hydra.main(config_path="conf/preprocess", config_name="config")
def myapp(config: DictConfig):
    analyze_fp(config)

if __name__=="__main__":
    myapp()