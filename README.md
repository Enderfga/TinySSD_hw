# Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection

This repo contains the code, data and trained models for the paper [Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection](https://arxiv.org/pdf/1802.06488.pdf).

## Quick Links

- [Overview](#overview)
- [How to Install](#how-to-install)
- [Description of Codes](#description-of-codes)
  - [Workspace](#workspace)
- [Preprocessing](#preprocessing)
  - [Preprocessed Data](#preprocessed-data)
  - [Generate Candidate Summaries](#generate-candidate-summaries)
  - [Preprocess Your Own Data](#preprocess-your-own-data)
- [How to Run](#how-to-run)
  - [Hyper-parameter Setting](#hyper-parameter-setting)
  - [Train](#train)
  - [Evaluate](#evaluate)
- [Results, Outputs, Checkpoints](#results-outputs-checkpoints)
- [Use BRIO with Huggingface](#use-brio-with-huggingface)

## Overview

Tiny SSD is a single-shot detection deep convolutional neural network for real-time embedded object detection.
It brings together the efficieny of Fire microarchitecture introduced in **SqueezeNet** and object detection performance of **SSD (Single Shot Object Detector)**.

![](https://img.enderfga.cn/img/ssd.svg)

![](https://img.enderfga.cn/img/image-20221018133431973.png)

## Requirements

* numpy
* pandas
* matplotlib
* torch
* torchvision

## How to Install

- ```shell
  conda create -n env python=3.8 -y
  conda activate env
  ```
- ```shell
  pip install -r requirements.txt
  ```

## Description of Codes

```
│──main.py                 -> Run models using different models
│──README.md
│──requirements.txt
│──test.py                 -> Testing Model
│──train.py                -> Training Model
│
├─data
│  │  dataloader.py         -> dataloader and transform
│  │  __init__.py
│  │
│  └─detection
│      │  create_train.py   -> data preprocessing
│      │
│      ├─background
│      ├─sysu_train
│      │  │  label.csv
│      │  │
│      │  └─images
│      ├─target
│      │      0.jpg
│      │      0.png
│      │      1.png
│      │
│      └─test
│              1.jpg
│              2.jpg
│
├─model
│  │  TinySSD.py             -> Definition of the model
│  │  __init__.py
│  │
│  └─checkpoints             -> Trained model weights
│          net_10.pkl
│          net_20.pkl
│          net_30.pkl
│          net_40.pkl
│          net_50.pkl
│
└─utils                      -> utility functions
        anchor.py
        iou.py
        utils.py
        __init__.py
```

## Preprocessing

We use /data/detection/background to generate the target detection dataset for our experiments.

### Preprocessed Data

```python
python data/detection/create_train.py
```

## How to Run

### Train

```console
python main.py --cuda --gpuid [list of gpuid] --config [name of the config (cnndm/xsum)] -l 
```

The checkpoints and log will be saved in a subfolder of `./cache`.

#### Example: training on CNNDM

```console
python main.py --cuda --gpuid 0 1 2 3 --config cnndm -l 
```

#### Finetuning from an existing checkpoint

```console
python main.py --cuda --gpuid [list of gpuid] -l --config [name of the config (cnndm/xsum)] --model_pt [model path]
```

model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

### Evaluate

For ROUGE calculation, we use the standard ROUGE Perl package from [here](https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5) in our paper. We lowercased and tokenized (using PTB Tokenizer) texts before calculating the ROUGE scores. Please note that the scores calculated by this package would be sightly *different* from the ROUGE scores calculated/reported during training/intermidiate stage of evalution, because we use a pure python-based ROUGE implementation to calculate those scores for better efficiency.

If you encounter problems when setting up the ROUGE Perl package (unfortunately it happens a lot :( ), you may consider using pure Python-based ROUGE package such as the one we used from the [compare-mt](https://github.com/neulab/compare-mt) package.

We provide the evaluation script in `cal_rouge.py`. If you are going to use Perl ROUGE package, please change line 13 into the path of your perl ROUGE package.

```python
_ROUGE_PATH = '/YOUR-ABSOLUTE-PATH/ROUGE-RELEASE-1.5.5/'
```

To evaluate the model performance, please first use the following command to generate the summaries.

```console
python main.py --cuda --gpuid [single gpu] --config [name of the config (cnndm/xsum)] -e --model_pt [model path] -g [evaluate the model as a generator] -r [evaluate the model as a scorer/reranker]
```

model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).
The output will be saved in a subfolder of `./result` having the same name of the checkpoint folder.

#### Example: evaluating the model as a generator on CNNDM

```console
# write the system-generated files to a file: ./result/cnndm/test.out
python main.py --cuda --gpuid 0 --config cnndm -e --model_pt cnndm/model_generation.bin -g

# tokenize the output file -> ./result/cnndm/test.out.tokenized (you may use other tokenizers)
export CLASSPATH=/your_path/stanford-corenlp-3.8.0.jar
cat ./result/cnndm/test.out | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ./result/cnndm/test.out.tokenized

# calculate the ROUGE scores using ROUGE Perl Package
python cal_rouge.py --ref ./cnndm/test.target.tokenized --hyp ./result/cnndm/test.out.tokenized -l

# calculate the ROUGE scores using ROUGE Python Implementation
python cal_rouge.py --ref ./cnndm/test.target.tokenized --hyp ./result/cnndm/test.out.tokenized -l -p
```

#### Example: evaluating the model as a scorer on CNNDM

```console
# rerank the candidate summaries
python main.py --cuda --gpuid 0 --config cnndm -e --model_pt cnndm/model_ranking.bin -r

# calculate the ROUGE scores using ROUGE Perl Package
# ./result/cnndm/reference and ./result/cnndm/candidate are two folders containing files. Each one of those files contain one summary
python cal_rouge.py --ref ./result/cnndm/reference --hyp ./result/cnndm/candidate -l

# calculate the ROUGE scores using ROUGE Python Implementation
# ./result/cnndm/reference and ./result/cnndm/candidate are two folders containing files. Each one of those files contain one summary
python cal_rouge.py --ref ./result/cnndm/reference --hyp ./result/cnndm/candidate -l -p
```

## Results, Outputs, Checkpoints

The following are ROUGE scores calcualted by the standard ROUGE Perl package.

### CNNDM

|                  | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ---------------- | ------- | ------- | ------- |
| BART             | 44.29   | 21.17   | 41.09   |
| BRIO-Ctr         | 47.28   | 22.93   | 44.15   |
| BRIO-Mul         | 47.78   | 23.55   | 44.57   |
| BRIO-Mul (Cased) | 48.01   | 23.76   | 44.63   |

### XSum

|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
| -------- | ------- | ------- | ------- |
| Pegasus  | 47.46   | 24.69   | 39.53   |
| BRIO-Ctr | 48.13   | 25.13   | 39.84   |
| BRIO-Mul | 49.07   | 25.59   | 40.40   |

### NYT

|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
| -------- | ------- | ------- | ------- |
| BART     | 55.78   | 36.61   | 52.60   |
| BRIO-Ctr | 55.98   | 36.54   | 52.51   |
| BRIO-Mul | 57.75   | 38.64   | 54.54   |

Our model outputs on these datasets can be found in `./output`.

We summarize the outputs and model checkpoints below.
You could load these checkpoints using `model.load_state_dict(torch.load(path_to_checkpoint))`.

|               | Checkpoints                                                                                                                                                                                                           | Model Output                                               | Reference Output                                             |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| CNNDM         | [model_generation.bin](https://drive.google.com/file/d/1CEBo6CCujl8QQwRKtYCMlS_s2_diBBS6/view?usp=sharing) `<br>` [model_ranking.bin](https://drive.google.com/file/d/1vxPBuTUvxYqARl9C4wegVVS9g5-h7cwO/view?usp=sharing) | [cnndm.test.ours.out](output/cnndm.test.ours.out)             | [cnndm.test.reference](output/cnndm.test.reference)             |
| CNNDM (Cased) | [model_generation.bin](https://drive.google.com/file/d/1YDUzNqbT6CC7VG3WfRspe2rM-j5DsjzT/view?usp=sharing)                                                                                                               | [cnndm.test.ours.cased.out](output/cnndm.test.ours.cased.out) | [cnndm.test.cased.reference](output/cnndm.test.cased.reference) |
| XSum          | [model_generation.bin](https://drive.google.com/file/d/135V7ybBGvjOVdTPuYA1R65uNAN_UoeSL/view?usp=sharing) `<br>` [model_ranking.bin](https://drive.google.com/file/d/1GX6EQcI222NXvvQ8Z0gKQPmc64podbeC/view?usp=sharing) | [xsum.test.ours.out](output/xsum.test.ours.out)               | [xsum.test.reference](output/xsum.test.reference)               |
