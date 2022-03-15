from tqdm import tqdm
from pathlib import Path
import random

import hydra
from omegaconf import DictConfig

def split_data(config):
    # Read utterance list file
    out_dir = Path(config.out_dir)
    with open(Path(config.out_dir) / "ipu.list", "r") as f:
        utt_list = [utt.strip() for utt in f]

    # Read person list file
    with open(Path(config.corpus_dir) / "speaker.list", "r") as f:
        person_id_list = [l.strip() for l in f if len(l.strip()) > 0]
        random.shuffle(person_id_list)

    # Split all data
    utt_list_train = []
    utt_list_dev = []
    utt_list_eval = []

    for person_id in tqdm(person_id_list, desc="all..."):
        utt_list_person = [utt for utt in utt_list if utt.startswith(f"{person_id}:")]
        n = len(utt_list_person)
        for phase in ["train", "dev", "eval"]:

            if phase == "train":
                utts_ = random.sample(utt_list_person, int(n*0.6))
                utt_list_train += utts_
                utt_list_person = [utt for utt in utt_list_person if not utt in utts_]
            elif phase == "dev":
                utts_ = random.sample(utt_list_person, int(n*0.2))
                utt_list_dev += utts_
                utt_list_person = [utt for utt in utt_list_person if not utt in utts_]
            elif phase == "eval":
                utt_list_eval += utt_list_person

    for phase, utt_list_phase in zip(
        ["train", "dev", "eval"],
        [utt_list_train, utt_list_dev, utt_list_eval]
    ):
        with open(out_dir / f"{phase}_all.list", "w") as f:
            f.write("\n".join(utt_list_phase))

    # Split group data
    group_persons_dict = {}
    with open(config.group_list_path, "r") as f:
        for l in f:
            i_class = int(l.strip().split(":")[1])
            if i_class in group_persons_dict.keys():
                group_persons_dict[i_class].append(l.strip().split(":")[0])
            else:
                group_persons_dict[i_class] = [l.strip().split(":")[0]]

    n_group = len(group_persons_dict.keys())
    for i in range(1, 1+n_group):
        person_id_list = group_persons_dict[i]

        utt_list_train = []
        utt_list_dev = []
        utt_list_eval = []
        for person_id in tqdm(person_id_list, desc=f"group{i}"):
            utt_list_person = [utt for utt in utt_list if utt.startswith(f"{person_id}:")]
            n = len(utt_list_person)
            for phase in ["train", "dev", "eval"]:

                if phase == "train":
                    utts_ = random.sample(utt_list_person, int(n*0.6))
                    utt_list_train += utts_
                    utt_list_person = [utt for utt in utt_list_person if not utt in utts_]
                elif phase == "dev":
                    utts_ = random.sample(utt_list_person, int(n*0.2))
                    utt_list_dev += utts_
                    utt_list_person = [utt for utt in utt_list_person if not utt in utts_]
                elif phase == "eval":
                    utt_list_eval += utt_list_person

        for phase, utt_list_phase in zip(
            ["train", "dev", "eval"],
            [utt_list_train, utt_list_dev, utt_list_eval]
        ):
            with open(out_dir / f"{phase}_group{i}.list", "w") as f:
                f.write("\n".join(utt_list_phase))

@hydra.main(config_path="conf/preprocess", config_name="config")
def myapp(config: DictConfig):
    split_data(config)

if __name__=="__main__":
    myapp()