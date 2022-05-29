# Transformer-based Keyword Spotting
Assignment 3: Keyword Spotting Using Transformer for EE 298Z Deep Learning (a.dulay)

## Table of contents
1. Intro
2. Usage Guide

## 1. Intro
![image](https://user-images.githubusercontent.com/43136926/170849779-acd83e41-a35d-438d-abac-52af76eeb56d.png)
<sup>The audio signal is converted to its mel spectrogram form as input for the model. In this sample, the leftmost image shows the waveform shape for a sample audio of "left" with the next few images being the signal visualized in the frequency spectrum. <a href="https://github.com/izzajalandoni/Deep-Learning-Helper/blob/main/Notes/Homework_3.pdf">More information regarding spectrograms and the image referenced can be found here.</a></sup>

Keyword spotting is used to detect key phrases or words in an audio stream. This can be used in edge devices to detect user commands similar to the way smart assistants such as Alexa, Siri, or Bixby function. In this repository, a transformer-based model is trained on the [Speech Commands dataset](https://arxiv.org/pdf/1804.03209.pdf). 

<br/>

| Data     | V1 Training | V2 Training |
|----------|-------------|-------------|
| **V1 Test**  | 85.4%       | 89.7%       |
| **V2 Test**  | 82.7%       | 88.2%       |

<sup>Table. 1 Baseline Top-One accuracy evaluations from the Speech Commands paper using a default convolution model</sup>

<br/>

A [dataloader used for KWS](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws_demo.ipynb) and a [transformer model](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/transformer/python/transformer_demo.ipynb) applied towards CIFAR 10 were modified to create the transformer KWS model. To test the model, modifications were done on the [kws-infer](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws-infer.py) to ensure compatibility.

For future work, the model performance may be improved further by adding more data from other datasets, resolving class imbalance by upsampling or synthesizing data from minority classes (e.g. "learn" and "backward" only contain 1,575 and 1,664 utterances respectively compared to the word "zero" which has 4,052 utterances in the dataset), and further experimentation with the parameter tuning. 

<br/>

|                             | ver. 1 | ver. 2 | ver. 3 | ver. 4 | ver. 5     |
|-----------------------------|:------:|:------:|:------:|:------:|:----------:|
| **Test Accuracy**               | 79.83% | 81.05% | 83.63% | 84.16% | **85.23%** |
| Batch Size                  | 64     | 32     | 64     | 32     | **64**     |
| Max Epochs                  | 60     | 45     | 70     | 70     | **20**     |
| Depth                       | 32     | 12     | 24     | 32     | **12**     |
| Embedding Dimension         | 128    | 64     | 128    | 64     | **128**    |
| Number of Heads             | 16     | 8      | 8      | 16     | **8**      |
| Number of Patches           | 4      | 8      | 4      | 4      | **4**      |
| Kernel Size                 | 3      | 3      | 3      | 3      | **3**      |
| Extra Fully Connected Layer | True   | False  | False  | True   | **False**  |

<sup>Table. 2 Parameters and corresponding test accuracy retrieved at the end of training from the implemented models. Parameters that were left unchanged across the different versions were not included on the table. It was observed that lowering some mel spectrogram variables caused zero values to show during computation. Additionally, increasing the MLP ratio, batch size, and FC layers tended to lower the test accuracy. However, these were observations done on a single GPU system and it is possible that with more resources and longer training time, the accuracy can converge to better results with the larger models.</sup>

## 2. Usage guide

1. Install dependencies

Properly setup Python3 and CUDA in your machine to leverage the GPU. Ensure installation of the following modules as well.

```bash
# Installs python libraries to emulate the same environment used for training.
pip install -r requirements.txt
```

```bash
# Done in general for torch usage; can be skipped if installing with requirements
pip install pytorch-lightning --upgrade
pip install torchmetrics --
```

```bash
# Important if kws-infer will be run. Installation of libasound2-dev & libportaudio2 is optional on Windows
sudo apt-get install libasound2-dev libportaudio2 
# Can be skipped if installing with requirements
pip install pysimplegui
pip install sounddevice 
pip install librosa
pip install validators
```

2. Train the model. Before running the training script, make sure that your machine has **at least 10GB** free due to the dataset that will be downloaded locally.

NOTE: This step is optional since the test script will use the trained model if there is no locally trained model.

```bash
python train.py
```

3. Run evaluation with python GUI
```bash
python kws-infer.py
```

4. Optionally, the initial training runs may also be checked under the `ntbks` folder for the different model versions specified in Table 2. These are standalone notebooks that can function when run on online platforms such as colab and kaggle.
