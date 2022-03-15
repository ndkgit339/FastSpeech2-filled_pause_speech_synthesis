# Speech synthesis with filled pauses and training scripts

This is a source implementation of speech synthesis model with filled pauses. There are...

## Requirements

You can install the Python requirements with

```
pip install -r requirements.txt
```

Our recommendation of the Python version is ``3.8``.

## Preparation

### BERT

Install BERT model to the directory ``./bert/`` from [here](https://nlp.ist.i.kyoto-u.ac.jp/?ku_bert_japanese). We use pytorch-pretrained-BERT with LARGE WWM version.

## Synthesize speech with filled pauses

This consists of two processes: filled pause prediction and speech synthesis.

### Step 0: Preparation

First of all, prepare filled pause prediction model and speech synthesis model. You can train a prediction model in [filledpause_prediction_group](https://github.com/ndkgit339/filledpause_prediction_group) and a speech synthesis model by using the script described below. Put the prediction model in ``./xxx`` and the speech synthesis model in ``./xxx``. You can see examples there.

### Step 1: Prediction

1. First, prepare a file of the utterance list. You can see an example of that in ``./predict_preprocessed_data/example``.
2. Next, run the script of preprocess for filled pause prediction. This follows the setting written in ``./config_predict/preprocess.yaml``. Change the setting accordingly.

```
python predict_preprocess.py
```

3. Next, run the script of filled pause prediction. This follows the setting written in ``./config_predict/predict.yaml``. Change the setting accordingly.

```
python predict_fp.py
```

4. Finally, run the script of postprocess as the preparation for the next synthesis step. This follows the setting written in ``./config_predict/postprocess.yaml``. Change the setting accordingly.

```
python predict_postprocess.py
```

### Step 2: Synthesis



## Train a speech synthesis model

### Step 1: Preparation

1. First, put files of phoneme labels, accents, and filled pause tags into the directory of preprocessed data ``./preprocessed_data``, and put files of raw texts and waves into the directory of raw data ``./raw_data``. You can see an example with the required formats thre.
   
2. Generate TextGrid files following an open-sourced script, [TextGridConverter](https://github.com/Syuparn/TextGridConverter).
   
3. The required directory structure is as follows:
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
The script ``preprocess.py`` ... 
This follows the setting written in ``conf/xxx.yaml``. Change the setting accordingly.
```
python preprocess.py
```

### Step 3: Training
The script ``train.py`` train the xxx. This follows the setting written in ``conf/train/config.yaml``. Change the setting accordingly.
```
python train.py
```
You can select whether you use FP tag or not in ``conf/train/config.yaml``.
```
use_fp_tag: True or False
```

## References
- 

- 

## Contributors
- [Yuta Matsunaga](https://sites.google.com/g.ecc.u-tokyo.ac.jp/yuta-matsunaga/home) (The University of Tokyo, Japan) [main contributor]
- [Takaaki Saeki](https://takaaki-saeki.github.io/) (The University of Tokyo, Japan)
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (The University of Tokyo, Japan)
- [Hiroshi Saruwatari](https://researchmap.jp/read0102891/) (The University of Tokyo, Japan)

## Citation
```
Coming soon...
```

# FastSpeech2 JSUT implementation (Scroll down for original readme)
## How To setup and start training
### Download JSUT
JSUT is available [here](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
### setup enviornmet and do preprocessing and training
change jsut path in retrieve_transcripts.py to where your jsut folder is 
```
git submodule update --init
pipenv --python 3
pipenv install
unzip hifigan/generator_universal.pth.tar.zip -d hifigan/

mkdir -p raw_data/JSUT/JSUT
cp path/to/JSUT/*/wav/*.wav raw_data/JSUT/JSUT
python retrieve_transcripts.py
python prepare_tg_accent.py jsut-lab/ preprocessed_data/JSUT/ JSUT --with_accent True
python3 preprocess.py config/JSUT/preprocess.yaml #this may take some time
python train.py -p config/JSUT/preprocess.yaml -m config/JSUT/model.yaml -t config/JSUT/train.yaml
```
## Synthesize Japanese
Synthesis Example
```
python3 synthesize.py --text "音声合成、たのちい" --speaker_id 0 --restore_step 20000 --mode single -p config/JSUT/preprocess.yaml -m config/JSUT/model.yaml -t config/JSUT/train.yaml
```

# FastSpeech 2 - PyTorch Implementation

This is a PyTorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1). 
This project is based on [xcmyz's implementation](https://github.com/xcmyz/FastSpeech) of FastSpeech. Feel free to use/modify the code.

There are several versions of FastSpeech 2.
This implementation is more similar to [version 1](https://arxiv.org/abs/2006.04558v1), which uses F0 values as the pitch features.
On the other hand, pitch spectrograms extracted by continuous wavelet transform are used as the pitch features in the [later versions](https://arxiv.org/abs/2006.04558).

![](./img/model.png)

# Updates
- 2021/2/26: Support English and Mandarin TTS
- 2021/2/26: Support multi-speaker TTS (AISHELL-3 and LibriTTS)
- 2021/2/26: Support MelGAN and HiFi-GAN vocoder

# Audio Samples
Audio samples generated by this implementation can be found [here](https://ming024.github.io/FastSpeech2/). 

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them in ``output/ckpt/LJSpeech/`` or ``output/ckpt/AISHELL3``.

For English single-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

For Mandarin multi-speaker TTS, try
```
python3 synthesize.py --text "大家好" --speaker_id SPEAKER_ID --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

The generated utterances will be put in ``output/result/``.

Here is an example of synthesized mel-spectrogram of the sentence "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", with the English single-speaker TTS model.  
![](./img/synthesized_melspectrogram.png)

## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
to synthesize all utterances in ``preprocessed_data/LJSpeech/val.txt``

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control 0.8 --energy_control 0.8
```

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [AISHELL-3](http://www.aishelltech.com/aishell_3): a Mandarin TTS dataset with 218 male and female speakers, roughly 85 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.

We take LJSpeech as an example hereafter.

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments for the LJSpeech and AISHELL-3 datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).
You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Alternately, you can align the corpus by yourself. 
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## Training

Train your model with
```
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

The model takes less than 10k steps (less than 1 hour on my GTX1080Ti GPU) of training to generate audio samples with acceptable quality, which is much more efficient than the autoregressive models such as Tacotron2.

# TensorBoard

Use
```
tensorboard --logdir output/log/LJSpeech
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Implementation Issues

- Following [xcmyz's implementation](https://github.com/xcmyz/FastSpeech), I use an additional Tacotron-2-styled Postnet after the decoder, which is not used in the original paper.
- Gradient clipping is used in the training.
- In my experience, using phoneme-level pitch and energy prediction instead of frame-level prediction results in much better prosody, and normalizing the pitch and energy features also helps. Please refer to ``config/README.md`` for more details.

Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model.

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation
```
@misc{chien2021investigating,
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  author={Chung-Ming Chien and Jheng-Hao Lin and Chien-yu Huang and Po-chun Hsu and Hung-yi Lee},
  year={2021},
  eprint={2103.04088},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```
