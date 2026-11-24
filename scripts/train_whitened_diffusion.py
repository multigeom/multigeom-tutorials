"""
Train the diffusion model of notebook 16 (step 2): the model of notebook 8's type (the
diffusion config of the inception V1 model, the paper's recipe), trained without
responses on the TinyImageNet training images whitened by the filter of notebook 16
(Olshausen & Field, data/02_intermediate/16_whitening/filter.pt).

Run from the repository root, in the background (about 5 h on one GPU; it resumes from
last.pt if interrupted):

    nohup python scripts/train_whitened_diffusion.py > train_whitened.out 2>&1 &
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from multigeom.config import load_config
from multigeom.datasets import CANONICAL_TRANSFORM, TinyImageNetDataset, WhiteningFilter
from multigeom.diffusion.train import train_unconditional

FILTER = Path("data/02_intermediate/16_whitening/filter.pt")
OUT_DIR = Path("runs/diffusion/unconditional/tinyimagenet-whitened")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=None, help="default: the recipe")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # the recipe of the V1 model (its 50 neurons give the width of the null token)
    cfg = load_config("diffusion")
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    # the training images of the V1 model, whitened by the filter of notebook 16
    start = time.time()
    train = TinyImageNetDataset(
        split=cfg.train.data_split, transform=CANONICAL_TRANSFORM
    )
    train.image_paths = train.image_paths[: cfg.train.n_train_images]
    images = torch.stack([train[index] for index in range(len(train))])
    whitening = WhiteningFilter.from_state_dict(torch.load(FILTER))
    images = torch.cat([whitening(batch) for batch in images.split(2000)])
    print(
        f"{len(images)} images whitened ({whitening.state_dict()}), pixel std"
        f" {images.std():.3f}, in {time.time() - start:.0f} s",
        flush=True,
    )

    # the model, trained on the null token only
    train_unconditional(
        cfg,
        images,
        args.out_dir,
        num_workers=args.num_workers,
        resume=True,
        progress=False,
        log=lambda message: print(f"{time.strftime('%H:%M:%S')} {message}", flush=True),
        metadata={
            "encoder_name": "unconditional",
            "area": "tinyimagenet-whitened",
            "whitening": whitening.state_dict(),
        },
    )


if __name__ == "__main__":
    main()
