# Cross-Modal Self-Attention with Multi-Task Pre-Training for Medical Visual Question Answering [paper](https://www.researchgate.net/publication/351229736_Cross-Modal_Self-Attention_with_Multi-Task_Pre-Training_for_Medical_Visual_Question_Answering#fullTextFileContent) [ICMR 2021 Best Poster Paper Award!](icmr2021.org/awards.html)

This repository is the official implementation of `CMSA-MTPT` for the visual question answering task in medical domain. Our model achieved **56.1** for open-ended and **77.3** for close-end on [VQA-RAD dataset](https://www.nature.com/articles/sdata2018251#data-citations). Up to 2021-5-28, the proposed models achieves the `SOTA` on the VQA-RAD dataset. For the detail, please refer to [link](https://www.researchgate.net/publication/351229736_Cross-Modal_Self-Attention_with_Multi-Task_Pre-Training_for_Medical_Visual_Question_Answering#fullTextFileContent).

The main contributer of this code is Guanqi Chen [link](https://github.com/chenguanqi). This repository is based on and inspired by @Jin-Hwa Kim's [work](https://github.com/jnhwkim/ban-vqa) and @Aizo-ai's [work](https://github.com/aioz-ai/MICCAI19-MedVQA). We sincerely thank for their sharing of the codes.


### Citation

Please cite this paper in your publications if it helps your research

```
@inproceedings{gong2021cross,
  author    = {Haifan Gong and
               Guanqi Chen and
               Sishuo Liu and
               Yizhou Yu and
               Guanbin Li},
  title     = {Cross-Modal Self-Attention with Multi-Task Pre-Training for Medical
               Visual Question Answering},
  booktitle = {{ICMR} '21: International Conference on Multimedia Retrieval, Taipei,
               Taiwan, August 21-24, 2021},
  pages     = {456--460},
  publisher = {{ACM}},
  year      = {2021},
  doi       = {10.1145/3460426.3463584},
}
```
## Note: You should replace the original imagenet pretrained encoder with the multi-task pretrained encoder in the drive or trained by yourself !!!

![Overview of cmsa-mtpt framework](overview.png)
Overview of the proposed medical VQA model. Our method consists of four components (with different colors in the figure): image feature extractor, question encoder, cross-modal self-attention (CMSA) module, and answer predictor.

![A novel multi-task pre-training framework](mtpt.png)

Multi-Task Pre-Training: the model is jointly trained with an image understanding task and a questionimage compatibility task. Depending on the dataset-specific image understanding task, the decoder can be selected as a fully convolutional network or a fully connected network.

### Prerequisites
torch                       1.0.1
torchvision                 0.4.0a0
numpy                       1.19.1

### Dataset and Pre-trained Models

The processed data should be downloaded via [link](https://pan.baidu.com/s/1MR81OMZLLIFHLyUcgiSbpA) with the extract code: `tkm8`. The downloaded file should be extracted to `data_RAD/` directory.
The pretrained models is available at [Baidu Drive](https://pan.baidu.com/s/1VQCAVADmrzEeRnW8GzsMfA) with extract code: `163k` Or [Google Drive](https://drive.google.com/drive/folders/1nlBaNwYtBK6Zmvsz7Yk7xzd3E9n-fQSC?usp=sharing)
The dataset for multi-task pretraining is available at [Baidu Drive](https://pan.baidu.com/s/1HuP7_PHRmkPCUPjQo5CE8Q) with extract code `gow6` Or [Google Drive](https://drive.google.com/file/d/1Y65SeAkc7gKjQfw0uUBisZAkXNQxgyHo/view?usp=sharing)

### Training and Testing
Just run the `train.sh` and the `test.sh` for training and evaluation.
The result json file can be found in the directory `results/`.

### Comaprison with the sota
![A novel multi-task pre-training framework](comparison_sota.png)


### License
MIT License

### More information
If you have any problem, no hesitate contact us at haifangong@outlook.com
HCP Lab Homepage: https://www.sysuhcp.com/
