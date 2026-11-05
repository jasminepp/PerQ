# PerQ: Inverse Generative Modeling for Neural Image Compression via Quantization Error Compensation

[Paper](https://arxiv.org/abs/2610.27374) · [Project page](https://jasminepp.github.io/PerQ/)

## Install

```bash
conda env create -f environment.yml
conda activate perq
bash setup_dcvc.sh          # DCVC-RT at a pinned commit, patched; builds its rANS entropy coder
bash setup_dcvc.sh --cuda   # optional: the DCVC-RT CUDA inference kernels
```

Run everything from the repository root. Real-bitstream coding needs a CUDA GPU.

## Data

Image folders under `datasets/` (symlinks are fine), holding only images: `MLIC` (the
[MLIC++ training set](https://huggingface.co/datasets/Whiteboat/MLIC-Train-100K)), `Kodak` (24 images) and
`CLIC2020-test` (428 images).

## Weights

Download both files into `weights/`:

| file | model cfg | md5 |
|---|---|---|
| [`dcvc-rt_backbone.pth`](https://drive.google.com/file/d/1vZhf5dvqtbotv0Mtksw9WbeEfyAgy2XV/view?usp=drive_link) (175 MB) | `cfgs/model/dcvc-rt_perq/dcvc-rt_intra.yaml` | `ead8ae81288aec5ce12c771075cc1df3` |
| [`perq_dcvc-rt.pth`](https://drive.google.com/file/d/1oVQ5ko1Qgug16GLEajc0fasIxlqcUyQT/view?usp=drive_link) (293 MB) | `cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml` | `906a78aaeae2c4f29b0c0a23d834bae9` |

`perq_dcvc-rt.pth` holds the frozen backbone and the compensator; `dcvc-rt_backbone.pth` holds the backbone alone, for the
backbone's own reconstruction.

## Training

```bash
# Stage 1: the backbone
accelerate launch --mixed_precision fp16 --dynamo_backend inductor train.py \
    --exp-cfg cfgs/exp/train/perq/mlic_rgb_64x1000k_r64_mse_lamb4-4096_lr2e-4.yaml \
    --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra.yaml
# Stage 2: the compensator and the discriminator on the frozen backbone
accelerate launch --mixed_precision fp16 --dynamo_backend inductor train.py \
    --exp-cfg cfgs/exp/train/perq/mlic_rgb_16x200k_r64_mse-elpips-s-1e0_lamb4-4096_lr1e-4.yaml \
    --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml \
    --pretrain outputs/mlic_rgb_64x1000k_r64_mse_lamb4-4096_lr2e-4_dcvc-rt_intra/checkpoints/training_stage_0/pytorch_model.bin
```

## Evaluation

```bash
python eval.py --exp-cfg cfgs/exp/train/perq/mlic_rgb_eval_r64_lamb4-4096_kodak_clic2020-test.yaml \
    --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml --pretrain weights/perq_dcvc-rt.pth
```

This codes real bitstreams on Kodak and CLIC2020-test at 8 rate points and writes
`outputs/<exp_name>_<model_name>/eval/results.json` with bpp, PSNR, MS-SSIM, LPIPS and DISTS per rate point, and FID
and KID on 256x256 patches. `results/` holds the Kodak and CLIC2020-test results of the paper's backbone and PerQ
models in this format.

One image:

```bash
python compress.py --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml --weights weights/perq_dcvc-rt.pth \
    --rp 31 --input image.png --output image.bin
python decompress.py --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml --weights weights/perq_dcvc-rt.pth \
    --input image.bin --output image_dec.png
```

`--rp` is the rate point, from 0 (lowest rate) to 63. `compress.py` writes the same bitstream with either model cfg,
and `decompress.py` decodes it with either: the backbone gives the MSE reconstruction, PerQ the perceptual one (each
with its own weights).

## Citation

```bibtex
@inproceedings{peng2026perq,
  title     = {PerQ: Inverse Generative Modeling for Neural Image Compression via Quantization Error Compensation},
  author    = {Peng, Tianhao and Kwan, Ho Man and Zhang, Fan and Liu, Shan and Bull, Dave},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

The backbone and its entropy coder are [DCVC-RT](https://github.com/microsoft/DCVC) (Jia et al., "Towards Practical
Real-Time Neural Video Compression", CVPR 2025), fetched and patched by `setup_dcvc.sh`. The training and evaluation
pipeline is derived from [NVRC](https://github.com/hmkx/NVRC) (Kwan et al., "NVRC: Neural Video Representation
Compression", NeurIPS 2024).