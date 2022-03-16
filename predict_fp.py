from pathlib import Path
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pytorch_lightning as pl

from filledpause_prediction_group.fp_pred_group.preprocessor import \
    process_morph, extract_feats_test
from filledpause_prediction_group.fp_pred_group.dataset import NoFPDataset
from filledpause_prediction_group.fp_pred_group.module import MyLightningModel


def preprocess(config: DictConfig):
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
    extract_feats_test(data_dir, config.fp_list_path, config.bert_model_dir, "utt_morphs")


def predict_utokyo_naist_lecture(data_dir, batch_size, num_workers, trainer, 
                                 model, out_dir, fps):

    # Paths
    in_feat_dir = Path(data_dir) / "infeats"
    out_feat_dir = Path(data_dir) / "outfeats"
    utt_list_path = Path(data_dir) / "utt.list"

    # Dataset
    in_feats_paths = list(in_feat_dir.glob("*-feats.npy"))
    out_feats_paths = [out_feat_dir / in_path.name for in_path in in_feats_paths]
    dataset = NoFPDataset(in_feats_paths, out_feats_paths, utt_list_path)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False
    )

    # output directory
    out_text_dir = out_dir / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    # Prediction
    out_utt_list = []
    prediction_list = []
    target_list = []
    outputs = trainer.predict(model, data_loader)
    for output in tqdm(outputs):
        batch_idx = output["batch_idx"]
        predictions = output["predictions"]
        targets = output["targets"]
        texts = output["texts"]

        for in_feats_path, prediction, text, target in zip(
            in_feats_paths[batch_idx*batch_size : (batch_idx+1)*batch_size],
            predictions,
            texts,
            targets
        ):
            prediction_list.append(prediction)
            target_list.append(target)

            text_len = len([t for t in text.split(" ") if len(t) > 0]) if text != "" else 0
            breath_para_name = in_feats_path.stem.replace("-feats", "")

            # prediction without true position
            filler_predictions = [int(i) for i in torch.argmax(prediction[:text_len+1], dim=1)]      
            i_utt = 0
            if filler_predictions[0] > 0:
                out_utt_list.append(f"{breath_para_name}-{i_utt}:(F)" + fps[filler_predictions[0]-1])
                with open(out_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
                    f.write(fps[filler_predictions[0]-1])
                i_utt += 1

            out_texts = []
            for t, f_pred in zip(
                [t for t in text.split(" ") if len(t) > 0], 
                filler_predictions[1:]
            ):
                out_texts.append(t)
                if f_pred > 0:
                    if len(out_texts) > 0:
                        out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
                        with open(out_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
                            f.write("".join(out_texts))
                        i_utt += 1

                    out_utt_list.append(f"{breath_para_name}-{i_utt}:(F)" + fps[int(f_pred)-1])
                    with open(out_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
                        f.write(fps[f_pred-1])
                    i_utt += 1

                    out_texts = []

            if len(out_texts) > 0:
                out_utt_list.append(f"{breath_para_name}-{i_utt}:" + "".join(out_texts))
                with open(out_text_dir / f"{breath_para_name}-{i_utt}.txt", "w") as f:
                    f.write("".join(out_texts))

    # Write predicted text
    out_utt_list = sorted(
        out_utt_list, 
        key=lambda u: tuple([int(n) for n in u.split(":")[0].split("-")])
    )
    with open(out_dir / "utt_filler_list.txt", "w") as f:
        f.write("\n".join(out_utt_list))

    # check
    with open(utt_list_path, "r") as f:
        utt_list = sorted(
            [l.strip() for l in f], 
            key=lambda u: tuple([int(n) for n in u.split(":")[:-1]])
        )
        utt_text = "".join([re.sub(r"\(F.*?\)", "", utt.split(":")[-1].replace(" ", "")) for utt in utt_list])
    out_utt_text = "".join([utt.split(":")[1] for utt in out_utt_list if not utt.split(":")[1].startswith("(F)")])
    assert utt_text == out_utt_text, f"utt_text should be equal to out_utt_text\nutt_text:\n{utt_text}\n\nout_utt_text:\n{out_utt_text}"

def predict(config: DictConfig):

    # Phase
    phase = "eval"

    # Out directory
    out_dir = Path(config[phase].out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # random seed
    pl.seed_everything(config.random_seed)

    # filler list
    fp_list_path = Path(to_absolute_path(config.fp_list_path))
    with open(fp_list_path, "r") as f:
        fps = [l.strip() for l in f]

    # Load model
    model = hydra.utils.instantiate(config.model.netG)
    pl_model = MyLightningModel.load_from_checkpoint(
        config[phase].model_ckpt_path, model=model, fp_list=fps, strict=False)

    # Trainer
    trainer = pl.Trainer(gpus=config[phase].gpus,
                         auto_select_gpus=config[phase].auto_select_gpus)

    predict_utokyo_naist_lecture(config.data_dir, config.data.batch_size,
                                 config.data.num_workers, trainer, pl_model,
                                 out_dir, fps)


@hydra.main(config_path="predict_config", config_name="predict")
def main(config: DictConfig):

    print("--- Preprocess ---")
    preprocess(config)
    print("--- Predict ---")
    predict(config)


if __name__=="__main__":
    main()