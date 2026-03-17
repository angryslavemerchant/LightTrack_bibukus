"""
tracking/distill_xattn.py

Distillation training: pixel_corr_mat (teacher) → xattn (student).

Both teacher and student share the same frozen backbone + neck from a
pretrained LightTrackM checkpoint. Only the student's xattn layers train.

Cut points (both (B, 64, 16, 16), before CA):
  Teacher: pixel_corr_mat(zf_bn, xf_bn)          — raw dot-product correlation
  Student: post_dw(chan_proj(mha_out_reshaped))   — after the final 128→64 contraction

Loss: MSE between teacher and student outputs at the cut point.

Dataset is assumed to be pre-curated: two folders of aligned crops
(see prepare_coco_patches.py). DataLoader is imported from coco_patches_loader.py.
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
from lib.models.connect import PWCA, pixel_corr_mat
from lib.utils.utils import load_pretrain
from tracking.coco_patches_loader import build_loader

# =============================================================================
# CONFIG  — edit here
# =============================================================================

CHECKPOINT   = "../snapshot/LightTrackM/LightTrackM.pth"
TEMPLATE_DIR = "data/coco_patches/template"   # pre-cropped 128×128 patches
SEARCH_DIR   = "data/coco_patches/search"     # pre-cropped 256×256 patches (2.0× context)
SAVE_DIR     = "snapshot/LightTrackM"

PATH_NAME     = "back_04502514044521042540+cls_211000022+reg_100000111_ops_32"
SEARCH_SIZE   = 256
TEMPLATE_SIZE = 128
STRIDE        = 16
ADJ_CHANNEL   = 128
EMBED_DIM     = 96    # backbone output channels at DP stage
NUM_KERNEL    = 64    # hz*wz = 8*8 = correlation output channels
SEARCH_NUM    = 256   # hx*wx = 16*16

EPOCHS        = 20
BATCH_SIZE    = 32
LR            = 1e-3
NUM_WORKERS   = 4
LOG_INTERVAL  = 50    # steps between loss prints

# =============================================================================
# TEACHER
# =============================================================================

class TeacherNet(nn.Module):
    """
    Frozen teacher: backbone → neck → pixel_corr_mat cut.

    Output: (B, 64, 16, 16) raw dot-product correlation, before CA.
    All parameters frozen; always runs under torch.no_grad().
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        zf = self.backbone(z)               # (B, 96,  8,  8)
        xf = self.backbone(x)               # (B, 96, 16, 16)
        zf_bn, xf_bn = self.neck(zf, xf)   # BN_adj — separate BN per branch
        return pixel_corr_mat(zf_bn, xf_bn) # (B, 64, 16, 16)  ← cut point


def build_teacher(checkpoint: str) -> TeacherNet:
    """
    Load pretrained LightTrackM_Speed, extract backbone + neck, freeze everything.
    Returns a TeacherNet ready for inference.
    """
    base = LightTrackM_Speed(
        path_name=PATH_NAME,
        search_size=SEARCH_SIZE,
        template_size=TEMPLATE_SIZE,
        stride=STRIDE,
        adj_channel=ADJ_CHANNEL,
    )
    load_pretrain(base, checkpoint, print_unuse=False)

    teacher = TeacherNet(
        backbone=base.features,
        neck=base.neck,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher


# =============================================================================
# STUDENT
# =============================================================================

class StudentNet(nn.Module):
    """
    Trainable student: receives pre-computed (zf_bn, xf_bn) from the teacher's
    frozen backbone+neck, runs xattn correlation, returns features at the cut
    point — after the final 128→64 contraction in post_dw, before CA.

    Channel flow:
        zf_bn: (B, 96,  8,  8) — K/V
        xf_bn: (B, 96, 16, 16) — Q
        → MHA out:   (B, 256, 96)
        → reshape:   (B,  96, 16, 16)
        → chan_proj:  (B,  64, 16, 16)   [Conv2d 96→64]
        → post_dw[0]: DW + PW(64→128) + PW(128→64)
        → post_dw[1]: DW + PW(64→128) + PW(128→64)  ← cut here
    """

    def __init__(self, pwca: PWCA):
        super().__init__()
        self.p = pwca

    def forward(self, zf_bn: torch.Tensor, xf_bn: torch.Tensor) -> torch.Tensor:
        p = self.p
        b, c, hz, wz = zf_bn.size()
        hx, wx = xf_bn.size(2), xf_bn.size(3)

        x_seq = xf_bn.flatten(2).permute(0, 2, 1)              # (B, 256, 96)  Q
        z_seq = zf_bn.flatten(2).permute(0, 2, 1)              # (B,  64, 96)  K/V

        mha_out = p.mha(
            p.q_proj(x_seq),
            p.k_proj(z_seq),
            p.v_proj(z_seq),
        )[0]                                                    # (B, 256, 96)

        corr = mha_out.permute(0, 2, 1).reshape(b, -1, hx, wx) # (B, 96, 16, 16)
        corr = p.chan_proj(corr)                                 # (B, 64, 16, 16)
        corr = p.post_dw(corr)                                  # (B, 64, 16, 16) ← cut
        return corr


def build_student() -> StudentNet:
    """Fresh PWCA(xattn) wrapped in StudentNet. All parameters trainable."""
    pwca = PWCA(
        num_channel=NUM_KERNEL,
        CA=True,          # CA is built but not called at the cut point
        corr_type="xattn",
        embed_dim=EMBED_DIM,
        search_num=SEARCH_NUM,
    )
    return StudentNet(pwca)


# =============================================================================
# TRAINING
# =============================================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = build_loader(TEMPLATE_DIR, SEARCH_DIR, BATCH_SIZE, NUM_WORKERS)
    print(f"Dataset: {len(loader.dataset)} pairs, {len(loader)} batches/epoch")

    teacher = build_teacher(CHECKPOINT).to(device)
    student = build_student().to(device)

    trainable = list(student.parameters())
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
        epoch_loss = 0.0

        step_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for step, (z_batch, x_batch) in enumerate(step_bar):
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)

            with torch.no_grad():
                zf = teacher.backbone(z_batch)          # (B, 96, 8, 8)
                xf = teacher.backbone(x_batch)          # (B, 96, 16, 16)
                zf_bn, xf_bn = teacher.neck(zf, xf)
                teacher_out = pixel_corr_mat(zf_bn, xf_bn)  # (B, 64, 16, 16)

            student_out = student(zf_bn, xf_bn)             # (B, 64, 16, 16)
            loss = F.mse_loss(student_out, teacher_out)

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
        ckpt_path = os.path.join(SAVE_DIR, f"xattn_epoch{epoch:03d}.pth")
        torch.save(student.p.state_dict(), ckpt_path)

        liveloss.update({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})
        liveloss.send()

        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")

    print("Done.")


if __name__ == "__main__":
    train()
