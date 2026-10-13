# CausticsNet and BackscatterNet

Re-implementation of [Self-Supervised Underwater Caustics Removal and Descattering via Deep Monocular SLAM](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11219.pdf) (Jonathan Sauder and Devis Tuia, ECCV 2024). This is not the code that produced the numbers in the paper. It trains the same two networks on top of the frozen [DeepReefMap](https://github.com/eceo-epfl/deepreefmap) SC-SfMLearner, and it was written with heavy use of coding agents.

Project page: [josauder.github.io/backscatternet_causticsnet](https://josauder.github.io/backscatternet_causticsnet/). Implementation details are in [Appendix A](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11219-supp.pdf).

## Install

Python 3.10–3.12. [uv](https://docs.astral.sh/uv/) installs the package and the DeepReefMap git dependency:

```bash
uv sync
```

## Run on a video

Download the weights and a short demo clip from the [v1.0 release](https://github.com/josauder/backscatternet_causticsnet/releases/tag/v1.0), then write a six-panel mp4:

```bash
curl -L -o causticsnet.pt https://github.com/josauder/backscatternet_causticsnet/releases/download/v1.0/causticsnet.pt
curl -L -o backscatternet.pt https://github.com/josauder/backscatternet_causticsnet/releases/download/v1.0/backscatternet.pt
curl -L -o demo.mp4 https://github.com/josauder/backscatternet_causticsnet/releases/download/v1.0/demo.mp4

uv run python scripts/infer_video.py demo.mp4 -o panels.mp4 \
  --caustics causticsnet.pt \
  --backscatter backscatternet.pt
```

The panels are the input, caustics removed, both effects removed, the estimated caustics, the estimated backscatter, and depth. The depth network is downloaded from `EPFL-ECEO/deepreefmap-sfm-net` on first use.

The default `--camera-profile gopro_hero_10` undistorts 1920×1080 GoPro Hero 10 footage before resizing, which is how the training frames were prepared. Other resolutions need `--camera-profile none`, or another DeepReefMap profile that matches the video.

## Checkpoints and the demo video

`causticsnet.pt`, `backscatternet.pt`, and `demo.mp4` are assets on the GitHub release above. They are not in the git tree, so cloning the repository does not download them. Each asset is a normal browser download, or the `curl` commands in the previous section.

## CausticsNet target

CausticsNet is trained to predict a residual `R = I_i - warp(I_j → i)`. `R` is detached, so it is a fixed regression target and the squared error updates only the prediction `C`. The self-discovered depth-consistency mask and the stationary auto-mask weight that squared error, `(C - R)²`. They do not multiply `R`, and they do not change the value `C` is compared against. The segmentation mask is the exception: where the view's own weight is 0, the target is set to 0 before the detach.

## Differences from the paper

- **Camera.** Frames are undistorted up front with DeepReefMap's rectifier, then warped with a linear pinhole `K`. The paper uses a universal camera model with a learned distortion, which converged to −0.4407, and does not pre-rectify.
- **Geometry network.** Depth and pose come from the frozen DeepReefMap SC-SfMLearner: a ResNet-18 disparity network and a separate ResNet-18 pose network, at 512×256, with inputs normalized as `(x − 0.45) / 0.225`. The paper trains its own SC-SfMLearner (ResNeXt-50 encoder, DeepLabV3+ depth decoder, pose decoder on concatenated features) for 2 million iterations at 608×352.
- **CausticsNet frame gap.** Pairs are drawn uniformly from 2–5 frames at 20 fps (0.10–0.25 s). Appendix A.2 uses 3–6 frames (0.15–0.30 s).
- **BackscatterNet frame gap.** Pairs are drawn uniformly from 3–20 frames (0.15–1.00 s). Appendix A.3 uses 3–25 frames (0.15–1.25 s).
- **Pose chaining.** A gap is split into hops of at most 4 frames (`max_hop: 4`). The paper composes poses through every second frame.
- **Schedule.** CausticsNet runs for 50k steps at batch 128. BackscatterNet runs for 10k steps at batch 32. The paper trains each network for 400k iterations and does not state the batch size. The optimizer matches the paper: Adam, learning rate 2e-4, BatchNorm momentum 0.01. The trained networks use bf16 here.
- **Moving-object mask.** CoralscapesV2 labels set the loss weight to zero on human, transect tools, fish, and transect line. The paper does not do this. Appendix B.4 lists masking those classes as a possible later fix.
- **Caustics residual.** See the paragraph above. Relative to Equation 7, both frames of a pair are supervised, and the squared error is weighted by the depth-consistency mask and the stationary auto-mask. The target itself is still the detached residual, or 0 where that view's own segmentation weight is 0.
- **Backscatter head.** Two conv–BN–ReLU layers (1×1, then 3×3) plus a zero-initialized 1×1 to six outputs, then a spatial mean. `B_inf` is a sigmoid starting at 0.3 and `beta` is a softplus starting at 1. Appendix A.3 describes three conv–BN–ReLU layers and does not specify this initialization.
- **R-CAUSTIC evaluation.** Images are resized to 256×512. Section 4.2 uses 352×576. The rest of the protocol matches the paper: bilinear upsampling, and a threshold on the LAB lightness difference between the input and the cleaned image, reported for one global threshold and for per-split thresholds.
- **Data.** Training uses a curated subset of rectified 1920×1080 reef video, resampled to 20 fps. A video contributes its transect windows, or the whole video when the annotation says it is roaming. Videos marked exclude, or upside down, are dropped. Section 4.1 uses 145 hours from 35 sites.
- **Experiments not included.** There is no COLMAP keypoint-matching comparison (Section 4.3) and no UIEB qualitative run (Appendix B.1).

What matches the paper: DeepLabV3+ with a ResNeXt-50 encoder for CausticsNet; a global six-parameter backscatter model; loss weights TV 1, ℓ2 0.05, clip 1 for CausticsNet and style 0.1, far-backscatter 0.5, clip 1 for BackscatterNet; the far prior on the furthest 5% of pixels, weighted by depth; the photometric term 0.15 L1 + 0.85 (1−SSIM)/2; and a frozen SLAM network while the restoration networks train.

A few choices are not stated in the paper. The encoders start from ImageNet weights. CausticsNet and BackscatterNet inputs use ImageNet normalization, while the frozen depth and pose network uses `(x − 0.45) / 0.225`. Color jitter is (0.1, 0.1, 0.1, 0.05), shared by both frames of a pair, with a second independent jitter for the style loss. Geometry is estimated on the un-jittered frames.

## Training

Training expects a directory of prepared frame sequences under `data/` (override with `CB_DATA_ROOT`). Checkpoints go to `runs/<run_name>/`.

```bash
uv run python -m caustics_backscatter.train --config configs/causticsnet.yaml wandb.mode=disabled
uv run python -m caustics_backscatter.train --config configs/backscatternet.yaml wandb.mode=disabled
```

`data/select_videos.py`, `data/prepare_frames.py`, and `data/segment_frames.py` build that directory from videos plus an annotation CSV. Pass the video directory and the CSV explicitly. Any config key can be overridden on the command line, for example `train.max_steps=50`.
