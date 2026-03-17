"""
tracking/distill_xattn.py

End-to-end output-level distillation: full model with pixel_corr_mat (teacher)
→ full model with xattn (student).

Frozen in both: backbone + neck (computed once, shared).
Trainable in student: feature_fusor (xattn pw_corr + adj_layer) + head (cls/reg towers).

Loss: MSE(s_cls, t_cls) + MSE(s_reg, t_reg)  — output level so towers adapt
freely to xattn features rather than trying to invert the softmax.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from livelossplot import PlotLosses

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _init_paths  # noqa

from lib.models.models import LightTrackM_Speed
from lib.models.connect import PWCA
from lib.utils.utils import load_pretrain
from tracking.coco_patches_loader import build_loader

# =============================================================================
# CONFIG  — edit here
# =============================================================================

CHECKPOINT    = "../snapshot/LightTrackM/LightTrackM.pth"
TEMPLATE_DIR  = "data/coco_patches/template"
SEARCH_DIR    = "data/coco_patches/search"
SAVE_DIR      = "snapshot/LightTrackM"

PATH_NAME     = "back_04502514044521042540+cls_211000022+reg_100000111_ops_32"
SEARCH_SIZE   = 256
TEMPLATE_SIZE = 128
STRIDE        = 16
ADJ_CHANNEL   = 128
EMBED_DIM     = 96
NUM_KERNEL    = 64
SEARCH_NUM    = 256

EPOCHS        = 20
BATCH_SIZE    = 32
LR            = 1e-4   # lower than before — towers are pretrained
NUM_WORKERS   = 4

# =============================================================================
# BUILD
# =============================================================================

def build_base() -> LightTrackM_Speed:
    model = LightTrackM_Speed(
        path_name=PATH_NAME,
        search_size=SEARCH_SIZE,
        template_size=TEMPLATE_SIZE,
        stride=STRIDE,
        adj_channel=ADJ_CHANNEL,
    )
    load_pretrain(model, CHECKPOINT, print_unuse=False)
    return model


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


# =============================================================================
# TRAINING
# =============================================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = build_loader(TEMPLATE_DIR, SEARCH_DIR, BATCH_SIZE, NUM_WORKERS)
    print(f"Dataset: {len(loader.dataset)} pairs, {len(loader)} batches/epoch")

    # --- teacher: full pretrained model, everything frozen ---
    teacher = build_base().to(device).eval()
    freeze(teacher)

    # --- student: same pretrained weights, swap pw_corr to xattn ---
    student = build_base().to(device)
    student.feature_fusor.pw_corr = PWCA(
        num_channel=NUM_KERNEL,
        CA=True,
        corr_type="xattn",
        embed_dim=EMBED_DIM,
        search_num=SEARCH_NUM,
    ).to(device)

    # freeze backbone + neck — only feature_fusor and head train
    freeze(student.features)
    freeze(student.neck)
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * len(loader),
        eta_min=LR * 0.01,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    liveloss = PlotLosses()

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs")
    for epoch in epoch_bar:
        student.train()
        # keep frozen parts in eval mode so BN stats don't drift
        student.features.eval()
        student.neck.eval()

        epoch_loss = 0.0

        step_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for step, (z_batch, x_batch) in enumerate(step_bar):
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)

            # shared backbone+neck forward — no grad, computed once
            with torch.no_grad():
                zf = student.features(z_batch)
                xf = student.features(x_batch)
                zf_bn, xf_bn = student.neck(zf, xf)

                # teacher output
                t_feat = teacher.feature_fusor(zf_bn, xf_bn)
                t_cls  = teacher.head(t_feat)["cls"]   # (B, 1, 16, 16)

            # student output (feature_fusor + head have grad)
            s_feat = student.feature_fusor(zf_bn, xf_bn)
            s_cls  = student.head(s_feat)["cls"]

            loss = F.mse_loss(s_cls, t_cls)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_loss = epoch_loss / len(loader)

        # save full student state dict so towers + fusor are captured together
        ckpt_path = os.path.join(SAVE_DIR, f"xattn_epoch{epoch:03d}.pth")
        torch.save(student.state_dict(), ckpt_path)

        liveloss.update({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})
        liveloss.send()
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")

    print("Done.")


if __name__ == "__main__":
    train()
