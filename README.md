# Cr-GAN

**Consistency-Regularized GAN for Few-Shot SAR Target Recognition**

[[Paper](https://arxiv.org/abs/2601.15681)]
[[Repository](https://github.com/yikuizhai/Cr-GAN)]

Authors: Yikui Zhai, Shikuang Liu, Wenlve Zhou, Hongsheng Zhang, Zhiheng Zhou,
Xiaolin Tian, and C. L. Philip Chen.

This is the minimal runnable research release of the DCGAN-based Cr-GAN
pipeline:

```text
few-shot data -> Cr-GAN training -> image generation -> SimCLR pre-training
              -> BIDFC-derived fine-tuning or linear evaluation
```

Only the code required by this pipeline is included. Datasets, generated
images, logs, experiment history, and checkpoints are intentionally excluded.
The BIDFC-derived component contains only the cleaned downstream fine-tuning
and linear-evaluation protocol, not the complete BIDFC pre-training project.

## Installation

```bash
python -m pip install -r requirements.txt
```

The pinned versions match the environment used for the release smoke tests.
Install the appropriate PyTorch/CUDA build for your system if necessary.

## Data layout

All commands expect the standard `ImageFolder` structure:

```text
dataset/
├── class_a/
│   ├── image_001.tif
│   └── image_002.tif
└── class_b/
    ├── image_001.tif
    └── image_002.tif
```

Prepare a deterministic few-shot subset:

```bash
python prepare_fewshot.py \
  --source /path/to/full/train \
  --output data/train_4shot \
  --shots 4 \
  --seed 10
```

## Train CR-GAN

```bash
python train.py \
  --path data/train_4shot \
  --log_dir runs/crgan \
  --classes 6 \
  --train_samples 24 \
  --gpu 0
```

## Generate images

```bash
python generate.py \
  --checkpoint runs/crgan/train_4shot/Checkpoints/G_iter_10001.pkl \
  --output outputs/generated \
  --num-images 5000 \
  --gpu 0
```

## SimCLR evaluation

The SimCLR implementation is retained under its original MIT license in
`evaluation/simclr/LICENSE.txt`.

```bash
python evaluation/simclr/run.py \
  --data outputs/generated \
  --dataset-name MSTAR \
  --output runs/simclr \
  --epochs 100 \
  --gpu-index 0
```

## Linear probe or fine-tuning

```bash
python evaluation/finetune/finetune.py \
  --train-dir data/train_4shot \
  --val-dir /path/to/validation \
  --pretrained runs/simclr/checkpoint_0100.pth.tar \
  --initialization simclr \
  --mode linear \
  --classes 6 \
  --output runs/finetune
```

Use `--mode finetune` to update the full ResNet-18 backbone. Reported metrics
are computed directly from validation predictions; no curve shaping or metric
post-processing is performed.

## Papers and acknowledgements

This release uses or adapts ideas and code associated with the following work:

- Cr-GAN: Y. Zhai, S. Liu, W. Zhou, H. Zhang, Z. Zhou, X. Tian, and
  C. L. P. Chen, "Consistency-Regularized GAN for Few-Shot SAR Target
  Recognition," arXiv:2601.15681, 2026.
  [[paper](https://arxiv.org/abs/2601.15681)]
- DCGAN: A. Radford, L. Metz, and S. Chintala, "Unsupervised Representation
  Learning with Deep Convolutional Generative Adversarial Networks," ICLR,
  2016. [[paper](https://arxiv.org/abs/1511.06434)]
- SimCLR: T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A Simple
  Framework for Contrastive Learning of Visual Representations," ICML,
  PMLR 119:1597-1607, 2020.
  [[paper](https://proceedings.mlr.press/v119/chen20j.html)]
- BIDFC: Y. Zhai et al., "Weakly Contrastive Learning via Batch Instance
  Discrimination and Feature Clustering for Small Sample SAR ATR," IEEE
  Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, article
  5204317, 2022, doi: 10.1109/TGRS.2021.3066195.
  [[paper](https://doi.org/10.1109/TGRS.2021.3066195)]
  [[code](https://github.com/Wenlve-Zhou/BIDFC-master)]

The bundled SimCLR implementation retains its original copyright notice and
MIT license in `evaluation/simclr/LICENSE.txt`.

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{zhai2026crgan,
  title   = {Consistency-Regularized GAN for Few-Shot SAR Target Recognition},
  author  = {Zhai, Yikui and Liu, Shikuang and Zhou, Wenlve and
             Zhang, Hongsheng and Zhou, Zhiheng and Tian, Xiaolin and
             Chen, C. L. Philip},
  journal = {arXiv preprint arXiv:2601.15681},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.15681}
}
```

## License

Cr-GAN is released under the MIT License. See `LICENSE` for details.
