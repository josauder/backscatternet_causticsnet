"""Sanity-check the frozen SC-SfMLearner geometry on the prepared frames.

For fixed clips at several frame gaps, reports the masked photometric error (Eq. 4) of
reprojecting one view into the other, for
  * input normalization: deepreefmap training ``(x-0.45)/0.225`` vs raw ``[0, 1]``,
  * poses: chained over <=3-frame hops vs one direct pose-network call,
next to the no-warp baseline (comparing the two frames directly). Lower is better; the
configuration used for training should clearly beat the baseline at every gap.

    python scripts/check_geometry.py --config configs/backscatternet.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from caustics_backscatter.config import load_config
from caustics_backscatter.data.dataset import ClipDataset, load_sequences
from caustics_backscatter.data.sampler import ClipSampler
from caustics_backscatter.geometry.warp import inverse_warp
from caustics_backscatter.losses.photometric import masked_mean, photometric_map
from caustics_backscatter.registry import GEOMETRY_PROVIDERS
from caustics_backscatter.tasks import Task  # noqa: F401  (registers components)
from caustics_backscatter.utils.viz import depth_to_uint8, signed_to_uint8, to_uint8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/backscatternet.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--gaps", type=int, nargs="+", default=[3, 6, 12, 24])
    parser.add_argument("--num-clips", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-viz", type=int, default=4)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provider = GEOMETRY_PROVIDERS.build(cfg.geometry).to(device)
    sequences = load_sequences(cfg.data.root, args.split)
    if not sequences:
        raise SystemExit(f"No finished '{args.split}' sequences under {cfg.data.root}")
    warp_kwargs = dict(cfg.warp)
    max_hop = cfg.geometry.max_hop

    import wandb

    out_dir = Path(cfg.paths.runs_root) / "check_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name="check_geometry", mode=cfg.wandb.mode, dir=str(out_dir))

    rows = []
    log: dict = {}
    for gap in args.gaps:
        for chained in (True, False):
            sampler = ClipSampler(num_views=2, gap_min=gap, gap_max=gap, max_hop=max_hop if chained else None, shuffle=False)
            dataset = ClipDataset(
                sequences, sampler, tuple(cfg.data.image_size), length=args.num_clips, deterministic=True, seed=cfg.seed
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
            for norm in ("train", "none"):
                provider.input_norm = norm
                warped_err = baseline_err = valid_frac = 0.0
                depths = []
                batches = 0
                viz = None
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    geom = provider(batch)
                    bsz = batch["frames"].shape[0]
                    views = batch["frames"][torch.arange(bsz, device=device)[:, None], batch["view_pos"]].float() / 255.0
                    for i, j in ((0, 1), (1, 0)):
                        warped, valid = inverse_warp(views[:, j], geom.depth[:, i], geom.T[:, j, i], geom.K, **warp_kwargs)
                        warped_err += float(masked_mean(photometric_map(views[:, i], warped), valid)) / 2
                        baseline_err += float(photometric_map(views[:, i], views[:, j]).mean()) / 2
                        valid_frac += float(valid.mean()) / 2
                        if viz is None:
                            viz = (views[:, i], views[:, j], warped * valid, (views[:, i] - warped) * valid, geom.depth[:, i])
                    depths.append(geom.depth.flatten())
                    batches += 1
                depth = torch.cat(depths).float()
                q = torch.quantile(depth[torch.randperm(depth.numel(), device=depth.device)[:1_000_000]],
                                   torch.tensor([0.05, 0.5, 0.95], device=depth.device))
                row = {
                    "gap": gap,
                    "poses": "chained" if chained else "direct",
                    "input_norm": norm,
                    "photometric_warped": warped_err / batches,
                    "photometric_no_warp": baseline_err / batches,
                    "valid_frac": valid_frac / batches,
                    "depth_p05": float(q[0]),
                    "depth_p50": float(q[1]),
                    "depth_p95": float(q[2]),
                }
                rows.append(row)
                print(json.dumps(row))
                key = f"gap{gap}/{row['poses']}/{norm}"
                tgt, src, warped, residual, depth_i = viz
                n = args.num_viz
                log[f"{key}/target"] = [wandb.Image(to_uint8(x)) for x in tgt[:n]]
                log[f"{key}/source"] = [wandb.Image(to_uint8(x)) for x in src[:n]]
                log[f"{key}/warped"] = [wandb.Image(to_uint8(x)) for x in warped[:n]]
                log[f"{key}/residual"] = [wandb.Image(signed_to_uint8(x, cfg.viz.residual_gain)) for x in residual[:n]]
                log[f"{key}/depth"] = [wandb.Image(depth_to_uint8(x)) for x in depth_i[:n]]

    log["summary"] = wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])
    wandb.log(log)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    wandb.finish()

    print(f"\n{'gap':>4} {'poses':>8} {'norm':>6} {'warped':>8} {'no_warp':>8} {'valid':>6} {'depth_p50':>9}")
    for r in rows:
        print(
            f"{r['gap']:>4} {r['poses']:>8} {r['input_norm']:>6} {r['photometric_warped']:>8.4f} "
            f"{r['photometric_no_warp']:>8.4f} {r['valid_frac']:>6.3f} {r['depth_p50']:>9.3f}"
        )


if __name__ == "__main__":
    main()
