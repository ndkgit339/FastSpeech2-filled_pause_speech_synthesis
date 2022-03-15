from pathlib import Path
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pytorch_lightning as pl


def pad_1d(x, max_len, constant_values=0):
    x = np.pad(
        x,
        (0, max_len - len(x)),
        mode="constant",
        constant_values=constant_values,
    )
    return x

def pad_2d(x, max_len, constant_values=0):
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x

class NoFillerDataset(Dataset):
    def __init__(self, in_paths, out_paths, utt_list_path=None):
        if utt_list_path is not None:
            self.text_dict = {}
            with open(utt_list_path, "r") as f:
                for l in f:
                    utt = l.strip()
                    if len(utt) > 0:
                        utt_name = "-".join(utt.split(":")[:-1])
                        text = " ".join([
                            w for w in utt.split(":")[-1].split(" ") 
                            if not w.startswith("(F")])
                        # text = re.sub(r"\(F.*?\)", "", utt.split(":")[-1])
                        self.text_dict[utt_name] = text

        self.in_paths = in_paths
        self.out_paths = out_paths

    def __getitem__(self, index):
        in_feat = np.load(self.in_paths[index]).astype(np.float32)
        out_feat = np.load(self.out_paths[index]).astype(np.float32)
        in_text = self.text_dict[self.in_paths[index].stem.replace("-feats", "")]
        sample = {
            "feat": in_feat, 
            "out_feat": out_feat, 
            "text": in_text,
        }
        return sample

    def __len__(self):
        return len(self.in_paths)

    def collate_fn(self, batch):
        lengths = [len(x["feat"]) for x in batch]
        max_len = max(lengths)
        x_batch = torch.stack([torch.from_numpy(pad_2d(x["feat"], max_len)) for x in batch])
        y_batch = torch.stack([torch.from_numpy(pad_1d(x["out_feat"], max_len)) for x in batch])
        text_batch = [x["text"] for x in batch]
        return x_batch, y_batch, text_batch


class MyLightningModel(pl.LightningModule):

    def __init__(
        self,
        model,
        train_filler_rate_dict=None,
        dev_filler_rate_dict=None,
        loss_weights=None,
        optimizer_name="Adam",
        optimizer_params=None,
        lr_scheduler_name="StepLR",
        lr_scheduler_params=None,
    ):

        super().__init__()

        self.model = model

        self.train_filler_rate_dict = train_filler_rate_dict
        self.dev_filler_rate_dict = dev_filler_rate_dict

        if loss_weights:
            self.criterion = nn.CrossEntropyLoss(torch.Tensor(loss_weights))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer_name=optimizer_name
        self.optimizer_params=optimizer_params
        self.lr_scheduler_name=lr_scheduler_name
        self.lr_scheduler_params=lr_scheduler_params

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, target = batch
        output = self.model(x)

        # Loss
        loss = self.criterion(output.transpose(1, -1), target.to(torch.long))

        # Logging
        train_logger = self.logger[0].experiment
        train_logger.add_scalar("Loss", loss, global_step=self.global_step)

        return {
            "loss": loss,
            "output": output.detach(),
            "target": target.detach(),
        }

    def validation_step(self, batch, batch_index):
        x, target = batch
        output = self.model(x)

        # Loss
        loss = self.criterion(output.transpose(1, -1), target.to(torch.long))
        self.log("val_loss", loss)

        return {
            "loss": loss,
            "output": output.detach(),
            "target": target.detach(),
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # this calls forward
        if len(batch) == 2:
            x, y = batch
            return {
                "predictions": self(x),
                "targets": y,
                "batch_idx": batch_idx,
            }
        elif len(batch) == 3:
            x, y, t = batch
            return {
                "predictions": self(x),
                "targets": y,
                "texts": t,
                "batch_idx": batch_idx,
            }

    def configure_optimizers(self):
        # Optimizer
        optimizer_class = getattr(optim, self.optimizer_name)
        optimizer = optimizer_class(
            self.parameters(), **self.optimizer_params
        )
        # lr scheduler
        lr_scheduler_class = getattr(optim.lr_scheduler, self.lr_scheduler_name)
        lr_scheduler = lr_scheduler_class(
            optimizer, **self.lr_scheduler_params
        )

        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def predict_utokyo_naist_lecture(
    config, trainer, model, out_dir, fps):

    # Paths
    in_feat_dir = Path(config.data.data_dir) / "infeats"
    out_feat_dir = Path(config.data.data_dir) / "outfeats"
    utt_list_path = Path(config.data.data_dir) / "utt.list"

    # Params
    batch_size = config.data.batch_size

    # Dataset
    in_feats_paths = list(in_feat_dir.glob("*-feats.npy"))
    out_feats_paths = [out_feat_dir / in_path.name for in_path in in_feats_paths]
    dataset = NoFillerDataset(in_feats_paths, out_feats_paths, utt_list_path)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=config.data.num_workers,
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

@hydra.main(config_path="config_predict", config_name="predict")
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
    filler_list_path = Path(to_absolute_path(config.data.filler_list))
    with open(filler_list_path, "r") as f:
        fps = [l.strip() for l in f]

    # Load model
    model = hydra.utils.instantiate(config.model.netG)
    pl_model = MyLightningModel.load_from_checkpoint(
        config[phase].model_ckpt_path, model=model, strict=False)

    # Trainer
    trainer = pl.Trainer(gpus=config[phase].gpus,
                         auto_select_gpus=config[phase].auto_select_gpus)

    predict_utokyo_naist_lecture(config, trainer, pl_model, out_dir, fps)

if __name__=="__main__":
    predict()