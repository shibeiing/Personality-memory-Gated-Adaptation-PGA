# Personality-memory Gated Adaptation: An Efficient Speaker Adaptation for Personalized End-to-end Automatic Speech Recognition
## Install
Our model and code are basicly based on Pytorch and ESPNet.
This repo only releases incremental modifications that are different from ESPNet 0.10.7a1.
Please first install them according to the versions in `requirements.txt`.

## Prosody Extractor
The pre-trained prosody extractor can be found in [prosody_extractor](https://github.com/shibeiing/Personality-aware-Training-PAT/tree/main/pretrained_models/prosody_extractor). 
- `model.acc.best` is the pretrained parameters, and the model definition refers `conv_san_model.py`.
- `model.json` is the configuration of prosody extractor at the training stage, you can find some hyper-parameter in it.

## Model definitions of PGA
- The model definition of PGA is provided in `espnet_model_with_pat_gating_parallel_adapter.py`. 
- The encoder equipped with parallel adapter is defined in `conformer_encoder_with_GatingPAdapter.py`.
- The parallel adapter is provided in `encoder_layer_with_gating_parallel_adapter.py`.

You can easily integrate it in your own model or other ESPNet ASR models.

## Checkpoints
- The checkpoint of generic ASR model is provided in `checkpoints/backbone`.
- The checkpoints of first training stage are provided in `checkpoints/stage1_adapter`.
- The checkpoints of second training stage are provided in `checkpoints/stage2_gated_finetune`.


## Dataset partition
We conduct experiments on KeSpeech and MagicData corpora. For KeSpeech, please refer the following paper:

```
Z. Tang, D. Wang, Y. Xu, J. Sun, and et.al., “Kespeech: An open source speech dataset of mandarin and its eight subdialects,” in NeurIPS Datasets and Benchmarks, 2021.
```

The data split for KeSpeech are provided in `kespeech_data_split`, you can split the original KeSpeech corpus into training, dev and test set according to the utterance id in the `*_text` files.

For the speaker adpation sets, we employ MagicData corpus. You can download it from the following URLs:
- https://magichub.com/datasets/sichuan-dialect-scripted-speech-corpus-daily-use-sentence/
- https://magichub.com/datasets/zhengzhou-dialect-scripted-speech-corpus-daily-use-sentence/

The data split of MagicData are provided in `adaption_spks`. There are six speakers from two accent domains (three speakers per domain).
For each speaker, we split the data into anchors and test cases, which are listed in `anchor_text` and `test_text`, respectively.

## License
This project is licensed under Apache-2.0 license.

## Citations

``` bibtex
@inproceedings{gu24b_interspeech,
  title     = {Personality-memory Gated Adaptation: An Efficient Speaker Adaptation for Personalized End-to-end Automatic Speech Recognition},
  author    = {Yue Gu and Zhihao Du and Shiliang Zhang and jiqing Han and Yongjun He},
  year      = {2024},
  booktitle = {{INTERSPEECH}},
  pages     = {2870--2874},
}
@inproceedings{gu2023pat,
  title={Personality-aware Training based Speaker Adaptation for End-to-end Speech Recognition},
  author={Yue Gu and Zhihao Du and Shiliang Zhang and Qian Chen and Jiqing Han},
  year={2023},
  booktitle={INTERSPEECH},
}
```
