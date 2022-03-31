# Speech synthesis with filled pauses and training scripts

This is an open-source implementation of a speech synthesis model with filled pauses based on FastSpeech2. The key charasteristics are as follows:
- [__Speech synthesis with filled pauses__](#speech-synthesis-with-filled-pauses) is a script to synthesize spontaneous speech with filled pauses from an input text without filled pauses using prepared models of filled pause prediction and speech synthesis.
- __Filled pause tags__ are introduced to the speech synthesis model to synthesize natural spontaneous speech with filled pauses. You can train a speech synthesis model that deals with filled pauses.

## Project page and audio samples

- Empirical study incorporating linguistic knowledge on filled pauses for personalized spontaneous speech synthesis (coming soon...)
- Audio samples are also available there.

## Requirements

You can install the Python requirements with

```
$ pip install -r requirements.txt
```

Our recommendation of the Python version is ``3.8``.

## Preparation

### BERT

Install BERT model to the directory ``./bert/`` from [here](https://nlp.ist.i.kyoto-u.ac.jp/?ku_bert_japanese). We use pytorch-pretrained-BERT with LARGE WWM version.

### Repository of filled pause prediction

Clone the repository of filled pause prediction ([filledpause_prediction_group](https://github.com/ndkgit339/filledpause_prediction_group)).

## Speech synthesis with filled pauses

You can synthesize spontaneous speech with filled pauses from an input text without filled pauses with prepared models of filled pause prediction and speech synthesis. This consists of two processes: filled pause prediction and speech synthesis.

### Step 0: Preparation of pre-trained models

First of all, prepare pre-trained filled pause prediction model and speech synthesis model. You can train a prediction model in [filledpause_prediction_group](https://github.com/ndkgit339/filledpause_prediction_group) and a speech synthesis model with the script described below. Write down tha path to the checkpoint file of the prediction model on ``./predict_config/predict.yaml``. Then, write down paths to configuration files (``model.yaml`` and ``train.yaml``) of the training of the synthesis model on ``./config/.../synthesize.yaml``. 

### Step 1: Prediction

1. First, prepare a file of the utterance list. You can see an example of that in ``./predict_data/example/preprocessed_data``.
2. Next, run the script of filled pause prediction. This follows the setting written in ``./config_predict/predict.yaml``. Change the setting accordingly.

```
$ python predict_fp.py
```

3. Finally, run the script of postprocess as the preparation for the next synthesis step. This follows the setting written in ``./config_predict/postprocess.yaml``. Change the setting accordingly.

```
$ python predict_postprocess.py
```

### Step 2: Synthesis

1. First, run the script of preprocess for speech synthesis. This follows the setting written in ``./config/.../preprocess.yaml``. Set the parameter ``test`` to ``True``. Change the setting accordingly.

```
$ python preprocess.py ${path to a preprocess configuration file}
```

2. Next, run the script of synthesis. This follows the setting written in ``./config/.../synthesize.yaml``. Change the setting accordingly.

```
$ python synthesize.py ${path to a synthesis configuration file} --restore_step ${numer of steps of the model used for synthesis}
```

## Train a speech synthesis model

You can train a spontaneous speech synthesis model including filled puases by fine-tuning a pretrained speech synthesis model trained on [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut). A pretrained model is available at ``./output/JSUT/ckpt/600000.pth.tar``.
This consists of three processed: preparation, preprocessing, and training. In the training, you can use a Japanese spontaneous speech corpus, [JLecSponSpeech](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/publications/spon_utokyo_lecture).

### Step 1: Preparation

1. First, prepare a pretrained speech synthesis model trained on [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut). A pretrained model is available at ``./output/JSUT/ckpt/600000.pth.tar``.

2. Put files of phoneme labels, accents, and filled pause tags into the directory of preprocessed data ``./preprocessed_data``, and put files of raw texts and waves into the directory of raw data ``./raw_data``. You can see an example with the required formats thre.
   
3. Generate TextGrid files following an open-source script ([TextGridConverter](https://github.com/Syuparn/TextGridConverter)).
   
4. The required directory structure is as follows:
```
|--- preprocessed_dir
|    |--- accent
|    |    └--- xxx.accent
|    |--- fp_tag
|    |    └--- speaker_name
|    |         └--- xxx.ftag
|    |--- TextGrid
|    |    └--- speaker_name
|    |         └--- xxx.TextGrid


|--- raw_dir
|    └--- speaker_name
|         |--- xxx.wav
|         └--- xxx.lab
```

### Step 2: Preprocess
Run the script ``preprocess.py`` to preprocess data for the training of the speech synthesis model.
- This follows the setting written in ``config/.../preprocess.yaml``. Set the parameter ``test`` to ``False``. Change the setting accordingly.
```
$ python preprocess.py path/to/preprocess/config
```

### Step 3: Training
Run the script ``train.py`` to train the speech synthesis model. You can use [JLecSponSpeech](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/publications/spon_utokyo_lecture).
- This follows the setting written in ``config/.../preprocess.yaml``, ``config/.../model.yaml``, and ``config/.../train.yaml``. Change the setting accordingly.
```
$ python train.py -p ${path to preprocess configuration file} -m ${path to model configuration file} -t ${path to training configuration file}
```
- You can select whether you use FP tag in ``config/.../train.yaml``.
```
use_fp_tag: True or False
```
- Write down the path to a checkpoint file of a pretrained model in ``config/.../train.yaml``.
```
fine_tune: True
fine_tune_ckpt_path: "./output/JSUT/ckpt/600000.pth.tar"
```

## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech 2 - PyTorch Implementation](https://github.com/ming024/FastSpeech2)
- [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
- [FastSpeech2 JSUT implementation](https://github.com/Wataru-Nakata/FastSpeech2-JSUT)
- [JLecSponSpeech](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/publications/spon_utokyo_lecture)
- [Japanese pretrained model of BERT](https://nlp.ist.i.kyoto-u.ac.jp/?ku_bert_japanese)
- [filledpause_prediction_group](https://github.com/ndkgit339/filledpause_prediction_group)
- [TextGridConverter](https://github.com/Syuparn/TextGridConverter)

## Contributors
- [Yuta Matsunaga](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/home) (The University of Tokyo, Japan) [main contributor]
- [Takaaki Saeki](https://takaaki-saeki.github.io/) (The University of Tokyo, Japan)
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (The University of Tokyo, Japan)
- [Hiroshi Saruwatari](https://researchmap.jp/read0102891/) (The University of Tokyo, Japan)

## Citation
```
Coming soon...