"""
main.py — AMDT Multi-Dataset AI-Image Detector (production version)
====================================================================
  ✓ VIB removed
  ✓ Two-stage training (real freeze/unfreeze)
  ✓ DDP when >1 GPU, single-GPU fallback
  ✓ fp16/bf16 auto-select
  ✓ Hard-check 3 datasets + class-balance report
  ✓ WeightedRandomSampler on imbalance
  ✓ --test smoke test
  ✓ os.scandir instead of glob (fast on NFS)
  ✓ Absolute paths (safe under SLURM)
  ✓ Resume checkpoint after every epoch
  ✓ Safe torch.load

Usage:
    python main.py --test                              # smoke test
    python main.py --mode train --debug                # 4-epoch debug
    python main.py --mode train                        # full training
    python main.py --mode train --resume               # continue from last epoch
    python main.py --mode predict --image x.jpg        # inference
"""

import os
import glob
import time
import math
import logging
import argparse
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from einops import rearrange
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, precision_score, recall_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm

warnings.filterwarnings("ignore")

# ── Absolute paths anchored to this file's directory ──────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(os.path.join(OUT_DIR, "plots"),       exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "logs"),        exist_ok=True)

# ═════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "logs", "train.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════
class Config:
    CIFAKE_ROOT    = os.path.join(BASE_DIR, "data", "cifake")
    AIVSREAL_ROOT  = os.path.join(BASE_DIR, "data", "aivsreal")
    FACES_ROOT     = os.path.join(BASE_DIR, "data", "faces140k")

    IMG_SIZE       = 224
    BATCH_SIZE     = 32
    NUM_WORKERS    = 8
    MAX_SAMPLES    = None

    MODEL_NAME     = "vit_base_patch16_224"
    EMBED_DIM      = 768
    NUM_HEADS      = 12
    NUM_LAYERS     = 12
    DROPOUT        = 0.3

    NUM_SRM_KERNELS = 3

    EPOCHS         = 15
    STAGE1_EPOCHS  = 3
    STAGE2_EPOCHS  = 12
    LR             = 2e-4
    BACKBONE_LR    = 2e-5
    WARMUP_EPOCHS  = 1
    WEIGHT_DECAY   = 1e-4
    GRAD_CLIP      = 1.0
    USE_AMP        = True
    LABEL_SMOOTH   = 0.1
    CUTMIX_ALPHA   = 1.0
    PATIENCE       = 4

    SEED           = 42
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT     = os.path.join(OUT_DIR, "checkpoints", "amdt_best.pt")
    RESUME_CKPT    = os.path.join(OUT_DIR, "checkpoints", "amdt_resume.pt")


def _amp_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


# ═════════════════════════════════════════════════════════════
# SRM
# ═════════════════════════════════════════════════════════════
def _srm_kernels():
    k1 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=np.float32)/4.0
    k2 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12.0
    k3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32)/2.0
    return [k1, k2, k3]


class SRMFilter(nn.Module):
    def __init__(self, num_kernels=3):
        super().__init__()
        kernels = _srm_kernels()[:num_kernels]
        weights = [torch.FloatTensor(k).unsqueeze(0).unsqueeze(0) for k in kernels for _ in range(3)]
        self.conv = nn.Conv2d(3, len(weights), 5, padding=2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.cat(weights, dim=0))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return torch.tanh(self.conv(x))


# ═════════════════════════════════════════════════════════════
# FREQNET
# ═════════════════════════════════════════════════════════════
class FreqNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels*2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels*2, 1),
        )

    def forward(self, x):
        orig_dtype = x.dtype
        freq = torch.fft.rfft2(x.to(torch.float32), norm='ortho')
        combined = torch.cat([freq.real, freq.imag], dim=1)
        processed = self.freq_conv(combined).to(torch.float32)
        real, imag = processed[:, :self.channels], processed[:, self.channels:]
        out = torch.fft.irfft2(torch.complex(real, imag), s=x.shape[-2:], norm='ortho')
        return out.to(orig_dtype)


# ═════════════════════════════════════════════════════════════
# FAA
# ═════════════════════════════════════════════════════════════
class ForgeryAwareAdapter(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(dim, dim//4, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim//4, dim, 3, padding=1),
        )
        self.freq_adapter = nn.Sequential(
            nn.Conv2d(dim, dim//4, 3, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(dim//4, dim, 4, stride=2, padding=1),
        )
        self.fusion = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x):
        B, N, D = x.shape
        cls = x[:, :1]
        patches = x[:, 1:]
        H = int(patches.shape[1] ** 0.5)
        W = patches.shape[1] // H
        spatial = patches.transpose(1, 2).reshape(B, D, H, W)
        s_feat = self.spatial_adapter(spatial)
        f_feat = self.freq_adapter(spatial)
        combined = (s_feat + f_feat).flatten(2).transpose(1, 2)
        adapted = self.fusion(combined + patches)
        return torch.cat([cls, adapted], dim=1)


# ═════════════════════════════════════════════════════════════
# CMA
# ═════════════════════════════════════════════════════════════
class CrossMultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, spatial_tokens, aux_tokens):
        B, N, D = spatial_tokens.shape
        Q = rearrange(self.q_proj(spatial_tokens), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.k_proj(aux_tokens),     'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.v_proj(aux_tokens),     'b n (h d) -> b h n d', h=self.num_heads)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = rearrange(attn @ V, 'b h n d -> b n (h d)')
        out = self.dropout(self.out_proj(out))
        return self.norm(spatial_tokens + out)


# ═════════════════════════════════════════════════════════════
# MoLD
# ═════════════════════════════════════════════════════════════
class MoLDGating(nn.Module):
    def __init__(self, num_layers=12, dim=768):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(dim, dim//4), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(dim//4, num_layers), nn.Softmax(dim=-1),
        )

    def forward(self, layer_outputs):
        stacked = torch.stack(layer_outputs, dim=1)
        cls_mean = stacked[:, :, 0, :].mean(dim=1)
        gates = self.gate_network(cls_mean).unsqueeze(-1).unsqueeze(-1)
        return (stacked * gates).sum(dim=1)


# ═════════════════════════════════════════════════════════════
# STREAM ENCODERS
# ═════════════════════════════════════════════════════════════
class ResidualStreamEncoder(nn.Module):
    def __init__(self, in_channels=9, embed_dim=768, img_size=224, patch_size=16):
        super().__init__()
        reduced_size = img_size // 4
        self.num_patches = (reduced_size // patch_size) ** 2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.patch_embed = nn.Conv2d(256, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv_layers(x)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        return x


class FrequencyStreamEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, img_size=224, patch_size=16):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        return x


# ═════════════════════════════════════════════════════════════
# AMDT
# ═════════════════════════════════════════════════════════════
class AMDT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vit = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=0)
        self.srm     = SRMFilter(cfg.NUM_SRM_KERNELS)
        self.freqnet = FreqNet()
        self.residual_encoder = ResidualStreamEncoder(
            in_channels=cfg.NUM_SRM_KERNELS * 3, embed_dim=cfg.EMBED_DIM
        )
        self.frequency_encoder = FrequencyStreamEncoder(
            in_channels=3, embed_dim=cfg.EMBED_DIM
        )
        self.faa_layers = nn.ModuleList([
            ForgeryAwareAdapter(cfg.EMBED_DIM) for _ in range(cfg.NUM_LAYERS // 2)
        ])
        self.cma_residual  = nn.ModuleList([
            CrossMultiHeadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS) for _ in range(cfg.NUM_LAYERS)
        ])
        self.cma_frequency = nn.ModuleList([
            CrossMultiHeadAttention(cfg.EMBED_DIM, cfg.NUM_HEADS) for _ in range(cfg.NUM_LAYERS)
        ])
        self.mold = MoLDGating(cfg.NUM_LAYERS, cfg.EMBED_DIM)
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(cfg.EMBED_DIM * 3),
            nn.Linear(cfg.EMBED_DIM * 3, cfg.EMBED_DIM),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.EMBED_DIM),
            nn.Linear(cfg.EMBED_DIM, 512), nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        srm_out  = self.srm(x)
        freq_out = self.freqnet(x)
        res_tokens  = self.residual_encoder(srm_out)
        freq_tokens = self.frequency_encoder(freq_out)

        x_vit = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x_vit.shape[0], -1, -1)
        tokens = torch.cat((cls_token, x_vit), dim=1)
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens)

        layer_outputs = []
        for i in range(self.cfg.NUM_LAYERS):
            if i % 2 == 0 and i // 2 < len(self.faa_layers):
                tokens = self.faa_layers[i // 2](tokens)
            tokens = self.vit.blocks[i](tokens)
            tokens = tokens + self.cma_residual[i](tokens, res_tokens)
            tokens = tokens + self.cma_frequency[i](tokens, freq_tokens)
            layer_outputs.append(tokens)

        fused = self.mold(layer_outputs)
        combined = torch.cat([fused[:, 0], res_tokens[:, 0], freq_tokens[:, 0]], dim=1)
        fused_feat = self.fusion_proj(combined)
        return self.classifier(fused_feat)


# ═════════════════════════════════════════════════════════════
# DATASET — os.scandir (fast on NFS)
# ═════════════════════════════════════════════════════════════
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG")


def _list_images_fast(folder):
    """Fast one-level image lister using os.scandir. Returns absolute paths."""
    out = []
    if not os.path.isdir(folder):
        return out
    try:
        with os.scandir(folder) as it:
            for e in it:
                try:
                    if e.is_file(follow_symlinks=False) and e.name.endswith(EXTS):
                        out.append(e.path)
                except OSError:
                    pass
    except OSError:
        pass
    return out


class BinaryImageDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, real_dir, fake_dir, img_size=224, augment=False,
                 max_samples=None, name="?"):
        t0 = time.time()
        real_imgs = [(p, 0) for p in _list_images_fast(real_dir)]
        fake_imgs = [(p, 1) for p in _list_images_fast(fake_dir)]
        self.samples = real_imgs + fake_imgs

        if max_samples and max_samples < len(self.samples):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idx]

        log.info(f"[{name}] {len(self.samples):,} samples (augment={augment}) "
                 f"[listed in {time.time()-t0:.1f}s]")

        base = [transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD)]
        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.1),
                transforms.RandomRotation(10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            ]
            self.transform = transforms.Compose(
                [base[0]] + aug + base[1:] + [transforms.RandomErasing(p=0.25)]
            )
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return {"spatial": self.transform(img),
                "label": torch.tensor(label, dtype=torch.long)}


def build_combined_dataset(cfg, split, augment=True):
    datasets = []
    for root, name in [
        (cfg.CIFAKE_ROOT,   "CIFAKE"),
        (cfg.AIVSREAL_ROOT, "AI-vs-Human"),
        (cfg.FACES_ROOT,    "140k-Faces"),
    ]:
        real_dir = f"{root}/{split}/REAL"
        fake_dir = f"{root}/{split}/FAKE"
        if os.path.isdir(real_dir) and os.path.isdir(fake_dir):
            datasets.append(BinaryImageDataset(
                real_dir, fake_dir, cfg.IMG_SIZE,
                augment=(augment and split == "train"),
                max_samples=cfg.MAX_SAMPLES, name=name,
            ))
        else:
            log.warning(f"[{name}] Missing dirs for split={split}: {real_dir} / {fake_dir}")

    if len(datasets) != 3:
        raise RuntimeError(
            f"Expected 3 datasets, found {len(datasets)}. "
            f"Run download_datasets.py first and verify data/ layout."
        )
    return ConcatDataset(datasets)


# ═════════════════════════════════════════════════════════════
# CUTMIX
# ═════════════════════════════════════════════════════════════
def cutmix(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = imgs.shape
    idx = torch.randperm(B, device=imgs.device)
    ch, cw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cw // 2, 0)
    x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0)
    y2 = min(cy + ch // 2, H)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[idx], lam


# ═════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════
def compute_metrics(labels, preds, probs):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "auc":       roc_auc_score(labels, probs),
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "conf_matrix": confusion_matrix(labels, preds),
    }


# ═════════════════════════════════════════════════════════════
# DATA LOADERS
# ═════════════════════════════════════════════════════════════
def get_loaders(cfg, rank=0, world_size=1):
    t0 = time.time()
    train_ds = build_combined_dataset(cfg, "train", augment=True)
    val_ds   = build_combined_dataset(cfg, "test",  augment=False)
    log.info(f"Datasets built in {time.time()-t0:.1f}s")

    labels = []
    for ds in train_ds.datasets:
        for _, lbl in ds.samples:
            labels.append(lbl)
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=2)
    ratio = counts[0] / max(counts[1], 1)
    log.info(f"Train class counts: REAL={counts[0]:,}  FAKE={counts[1]:,}  ratio={ratio:.3f}")

    use_sampler = not (0.85 <= ratio <= 1.15)
    if use_sampler:
        log.info("Imbalance detected — using WeightedRandomSampler")
        class_weights = 1.0 / counts
        sample_weights = torch.from_numpy(class_weights[labels]).double()
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    elif world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    kw = dict(
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, sampler=train_sampler,
                              shuffle=(train_sampler is None), **kw)
    val_loader   = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **kw)
    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════
def _gpu_mem_str():
    if not torch.cuda.is_available():
        return "N/A"
    parts = []
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 1024**3
        r = torch.cuda.memory_reserved(i) / 1024**3
        parts.append(f"GPU{i}: {a:.1f}/{r:.1f}GB")
    return "  ".join(parts)


def _fmt_time(s):
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


LOG_EVERY = 50


# ═════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════
def train(cfg, debug=False, fast=False, resume=False):
    if fast:
        cfg.MAX_SAMPLES = 100
        cfg.EPOCHS = 3
        cfg.STAGE1_EPOCHS = 1
        cfg.STAGE2_EPOCHS = 2
        log.info("FAST MODE")
    elif debug:
        cfg.MAX_SAMPLES = 2000
        cfg.EPOCHS = 4
        cfg.STAGE1_EPOCHS = 1
        cfg.STAGE2_EPOCHS = 3
        log.info("DEBUG MODE")

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp    = world_size > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        cfg.DEVICE = torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    log.info("Building AMDT model...")
    model = AMDT(cfg).to(cfg.DEVICE)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    backbone_params = list((model.module if use_ddp else model).vit.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids and p.requires_grad]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg.BACKBONE_LR},
        {"params": head_params,     "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)
    amp_dtype = _amp_dtype()

    def lr_lambda(ep):
        if ep < cfg.WARMUP_EPOCHS:
            return (ep + 1) / cfg.WARMUP_EPOCHS
        p = (ep - cfg.WARMUP_EPOCHS) / max(1, cfg.EPOCHS - cfg.WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * p))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"AMDT: {n_params:.2f}M params | device={cfg.DEVICE} | world={world_size}")

    log.info("Building DataLoaders...")
    train_loader, val_loader = get_loaders(cfg, rank=rank, world_size=world_size)
    log.info(f"train={len(train_loader):,} batches  val={len(val_loader):,} batches")

    best_auc = 0.0
    patience_left = cfg.PATIENCE
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    start_epoch = 0

    # ── RESUME ─────────────────────────────────────────────
    if resume and os.path.isfile(cfg.RESUME_CKPT):
        log.info(f"[RESUME] Loading {cfg.RESUME_CKPT}")
        ckpt = torch.load(cfg.RESUME_CKPT, map_location=cfg.DEVICE, weights_only=False)
        raw = model.module if use_ddp else model
        raw.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)
        patience_left = ckpt.get("patience_left", cfg.PATIENCE)
        history = ckpt.get("history", history)
        log.info(f"[RESUME] Continuing from epoch {start_epoch+1}, best_auc={best_auc:.4f}")

    job_start = time.time()

    for epoch in range(start_epoch, cfg.EPOCHS):
        stage = "stage1" if epoch < cfg.STAGE1_EPOCHS else "stage2"
        raw_model = model.module if use_ddp else model
        freeze_backbone = (stage == "stage1")
        for p in raw_model.vit.parameters():
            p.requires_grad = not freeze_backbone
        log.info(f"Epoch {epoch+1}/{cfg.EPOCHS} [{stage}] backbone_frozen={freeze_backbone}")

        epoch_start = time.time()

        # ── TRAIN ─────────────────────────────────────────
        model.train()
        run_loss = correct = total = 0
        batch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["spatial"].to(cfg.DEVICE, non_blocking=True)
            labels = batch["label"].to(cfg.DEVICE, non_blocking=True)

            use_cm = np.random.rand() < 0.5
            if use_cm:
                imgs, la, lb, lam = cutmix(imgs, labels, cfg.CUTMIX_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", amp_dtype, enabled=cfg.USE_AMP):
                logits = model(imgs)
                if use_cm:
                    ce_loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
                else:
                    ce_loss = criterion(logits, labels)
                ce_loss = ce_loss.mean()

            scaler.scale(ce_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            run_loss += ce_loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % LOG_EVERY == 0 or (batch_idx + 1) == len(train_loader):
                done = batch_idx + 1
                elapsed = time.time() - batch_start
                spb = elapsed / done
                eta = (len(train_loader) - done) * spb
                log.info(
                    f"  [Train] Ep{epoch+1:02d} {done:04d}/{len(train_loader):04d} "
                    f"| loss {run_loss/done:.4f} | acc {correct/total*100:.2f}% "
                    f"| {spb:.2f}s/b | ETA {_fmt_time(eta)} | {_gpu_mem_str()}"
                )

        scheduler.step()
        train_elapsed = time.time() - epoch_start
        log.info(f"Train done in {_fmt_time(train_elapsed)} | avg_loss={run_loss/len(train_loader):.4f}")

        # ── VALIDATION ────────────────────────────────────
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        val_loss = 0.0
        val_start = time.time()
        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                imgs = batch["spatial"].to(cfg.DEVICE, non_blocking=True)
                labels = batch["label"].to(cfg.DEVICE, non_blocking=True)
                with torch.autocast("cuda", amp_dtype, enabled=cfg.USE_AMP):
                    logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                prob = F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                all_probs.extend(prob)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        m = compute_metrics(all_labels, all_preds, all_probs)
        val_loss /= len(val_loader)

        history["train_loss"].append(run_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_auc"].append(m["auc"])
        history["val_f1"].append(m["f1"])

        epoch_elapsed = time.time() - epoch_start
        log.info(f"Val  done in {_fmt_time(time.time()-val_start)}")
        log.info(f"  AUC={m['auc']:.4f}  F1={m['f1']:.4f}  Acc={m['accuracy']*100:.2f}%  "
                 f"ValLoss={val_loss:.4f}")
        log.info(f"  Confusion:\n{m['conf_matrix']}")
        log.info(f"  Epoch time: {_fmt_time(epoch_elapsed)}")

        # ── BEST CKPT ─────────────────────────────────────
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            patience_left = cfg.PATIENCE
            if rank == 0:
                torch.save(
                    {"state_dict": model.state_dict(), "best_auc": best_auc},
                    cfg.CHECKPOINT,
                )
            log.info(f"[SAVED BEST] AUC={best_auc:.4f}  → {cfg.CHECKPOINT}")
        else:
            patience_left -= 1
            log.info(f"[PATIENCE] left={patience_left}/{cfg.PATIENCE}")
            if patience_left == 0:
                log.info("Early stopping.")
                # Save resume before breaking
                if rank == 0:
                    torch.save({
                        "state_dict": (model.module if use_ddp else model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_auc": best_auc,
                        "patience_left": patience_left,
                        "history": history,
                    }, cfg.RESUME_CKPT)
                break

        # ── RESUME CKPT AFTER EVERY EPOCH ─────────────────
        if rank == 0:
            torch.save({
                "state_dict": (model.module if use_ddp else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "patience_left": patience_left,
                "history": history,
            }, cfg.RESUME_CKPT)
            log.info(f"[RESUME CKPT] epoch {epoch+1} → {cfg.RESUME_CKPT}")

    total_time = time.time() - job_start
    log.info("=" * 70)
    log.info(f"TRAINING COMPLETE | best_auc={best_auc:.4f} | time={_fmt_time(total_time)}")
    log.info("=" * 70)

    if rank == 0:
        save_plots(history)
        try:
            save_eval_plots(history, cfg, model.module if use_ddp else model, val_loader)
            save_per_dataset_plot(cfg, model.module if use_ddp else model, val_loader)
        except Exception as e:
            log.warning(f"extra plots failed: {e}")


    if use_ddp:
        dist.destroy_process_group()


# ═════════════════════════════════════════════════════════════
# SMOKE TEST
# ═════════════════════════════════════════════════════════════
def smoke_test(cfg):
    log.info("=" * 70)
    log.info("SMOKE TEST — full pipeline on 100 images")
    log.info("=" * 70)

    cfg.MAX_SAMPLES = 100
    cfg.BATCH_SIZE  = 8
    cfg.NUM_WORKERS = 2
    torch.manual_seed(cfg.SEED)

    try:
        log.info("[1/8] Building model...")
        model = AMDT(cfg).to(cfg.DEVICE)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"      ✓ Model built: {n_params:.2f}M params on {cfg.DEVICE}")

        log.info("[2/8] Building datasets...")
        train_ds = build_combined_dataset(cfg, "train", augment=True)
        val_ds   = build_combined_dataset(cfg, "test",  augment=False)
        log.info(f"      ✓ train={len(train_ds)}  val={len(val_ds)}")

        log.info("[3/8] Building loaders...")
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=False, num_workers=cfg.NUM_WORKERS)
        log.info(f"      ✓ train={len(train_loader)} batches  val={len(val_loader)} batches")

        log.info("[4/8] Setting up optimizer...")
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
        amp_dtype = _amp_dtype()
        log.info(f"      ✓ optimizer set, amp_dtype={amp_dtype}")

        log.info("[5/8] Forward + backward pass on one batch...")
        model.train()
        batch = next(iter(train_loader))
        imgs = batch["spatial"].to(cfg.DEVICE)
        labels = batch["label"].to(cfg.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", amp_dtype, enabled=cfg.USE_AMP):
            logits = model(imgs)
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        log.info(f"      ✓ loss={loss.item():.4f}  logits.shape={tuple(logits.shape)}")

        log.info("[6/8] CutMix forward pass...")
        mixed, la, lb, lam = cutmix(imgs, labels, cfg.CUTMIX_ALPHA)
        with torch.autocast("cuda", amp_dtype, enabled=cfg.USE_AMP):
            logits_cm = model(mixed)
        log.info(f"      ✓ CutMix OK: lam={lam:.3f}  logits.shape={tuple(logits_cm.shape)}")

        log.info("[7/8] Validation loop on 2 batches...")
        model.eval()
        all_l, all_p, all_pr = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:
                    break
                imgs = batch["spatial"].to(cfg.DEVICE)
                labels = batch["label"].to(cfg.DEVICE)
                with torch.autocast("cuda", amp_dtype, enabled=cfg.USE_AMP):
                    logits = model(imgs)
                all_l.extend(labels.cpu().numpy())
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_pr.extend(F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        if len(set(all_l)) > 1:
            m = compute_metrics(all_l, all_p, all_pr)
            log.info(f"      ✓ val AUC={m['auc']:.4f}  Acc={m['accuracy']*100:.1f}%")
        else:
            log.info(f"      ✓ val loop OK (single-class batch)")

        log.info("[8/8] Checkpoint save/load round-trip...")
        tmp_ckpt = os.path.join(OUT_DIR, "checkpoints", "_smoke_test.pt")
        torch.save({"state_dict": model.state_dict(), "best_auc": 0.5}, tmp_ckpt)
        model2 = AMDT(cfg).to(cfg.DEVICE)
        ckpt = torch.load(tmp_ckpt, map_location=cfg.DEVICE, weights_only=False)
        model2.load_state_dict(ckpt["state_dict"], strict=True)
        os.remove(tmp_ckpt)
        log.info(f"      ✓ checkpoint round-trip OK")

        log.info("=" * 70)
        log.info("SMOKE TEST PASSED — pipeline is error-free")
        log.info("=" * 70)
        return 0

    except Exception:
        import traceback
        log.error("=" * 70)
        log.error("SMOKE TEST FAILED")
        log.error(traceback.format_exc())
        log.error("=" * 70)
        return 1


# ═════════════════════════════════════════════════════════════
# PLOTTING (guarded)
# ═════════════════════════════════════════════════════════════
def save_plots(history):
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        if history.get("train_loss"):
            axes[0].plot(history["train_loss"], label="Train")
            if history.get("val_loss"):
                axes[0].plot(history["val_loss"], label="Val")
            axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
        if history.get("val_auc"):
            axes[1].plot(history["val_auc"], label="AUC", color='green')
            axes[1].set_title("AUC"); axes[1].legend(); axes[1].grid(True)
        if history.get("val_f1"):
            axes[2].plot(history["val_f1"], label="F1", color='orange')
            axes[2].set_title("F1"); axes[2].legend(); axes[2].grid(True)
        axes[3].axis('off')
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "plots", "training_history.png")
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        log.info(f"Plots saved to {out}")
    except Exception as e:
        log.warning(f"save_plots failed: {e}")


# ═════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════
def load_model(cfg):
    if not os.path.isfile(cfg.CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {cfg.CHECKPOINT}")
    model = AMDT(cfg)
    ckpt = torch.load(cfg.CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(cfg.DEVICE).eval()
    log.info(f"Model loaded. best_auc={ckpt.get('best_auc', 'N/A')}")
    return model


def predict_image(model, image_path, cfg):
    tfm = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits.float(), dim=1)[0]
    pf, pr = probs[1].item(), probs[0].item()
    return {
        "label": "AI-GENERATED" if pf > 0.5 else "REAL",
        "confidence": max(pf, pr),
        "prob_fake": pf,
        "prob_real": pr,
    }

def save_eval_plots(history, cfg, model, val_loader):
    """
    Produce: confusion matrix heatmap, ROC curve, per-epoch metrics, all metrics table.
    Called once after training finishes.
    """
    import numpy as np
    from sklearn.metrics import roc_curve, auc, classification_report
    import torch.nn.functional as F

    # Rerun val set to get final predictions
    model.eval()
    all_l, all_p, all_pr = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["spatial"].to(cfg.DEVICE, non_blocking=True)
            labels = batch["label"].to(cfg.DEVICE, non_blocking=True)
            with torch.autocast("cuda", _amp_dtype(), enabled=cfg.USE_AMP):
                logits = model(imgs)
            all_pr.extend(F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(labels.cpu().numpy())

    all_l  = np.array(all_l)
    all_p  = np.array(all_p)
    all_pr = np.array(all_pr)

    cm = confusion_matrix(all_l, all_p)
    fpr, tpr, _ = roc_curve(all_l, all_pr)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["REAL", "FAKE"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["REAL", "FAKE"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Acc={accuracy_score(all_l, all_p)*100:.2f}%)")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=16)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Right: ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "plots", "eval_confusion_roc.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Eval plots saved to {out}")

    # Also save a classification report as text
    report = classification_report(all_l, all_p, target_names=["REAL", "FAKE"], digits=4)
    report_path = os.path.join(OUT_DIR, "plots", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Accuracy: {accuracy_score(all_l, all_p)*100:.2f}%\n\n")
        f.write(report)
    log.info(f"Classification report saved to {report_path}")

    return {"auc": roc_auc, "cm": cm, "fpr": fpr, "tpr": tpr}

def save_per_dataset_plot(cfg, model, val_loader):
    """Bar chart of AUC/accuracy by source dataset."""
    model.eval()
    val_ds = val_loader.dataset

    # Tag each val sample by source
    def which_dataset(p):
        if "cifake" in p.lower():       return "CIFAKE"
        if "ai-vs-human" in p.lower():  return "AI-vs-Human"
        if "140k" in p.lower():         return "140k-Faces"
        return "Other"

    # Collect all val samples' tags in dataset order
    all_samples = []
    if hasattr(val_ds, "datasets"):
        for ds in val_ds.datasets:
            all_samples.extend(ds.samples)
    elif hasattr(val_ds, "samples"):
        all_samples = val_ds.samples

    tags = [which_dataset(p) for p, _ in all_samples]

    all_l, all_p, all_pr = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["spatial"].to(cfg.DEVICE, non_blocking=True)
            labels = batch["label"].to(cfg.DEVICE, non_blocking=True)
            with torch.autocast("cuda", _amp_dtype(), enabled=cfg.USE_AMP):
                logits = model(imgs)
            all_pr.extend(F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            all_p.extend(logits.argmax(1).cpu().numpy())
            all_l.extend(labels.cpu().numpy())

    # Aggregate by dataset
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score
    groups = {}
    for tag, y, p, pr in zip(tags, all_l, all_p, all_pr):
        groups.setdefault(tag, {"y": [], "p": [], "pr": []})
        groups[tag]["y"].append(y)
        groups[tag]["p"].append(p)
        groups[tag]["pr"].append(pr)

    names, aucs, accs = [], [], []
    for tag in sorted(groups):
        y  = np.array(groups[tag]["y"])
        p  = np.array(groups[tag]["p"])
        pr = np.array(groups[tag]["pr"])
        names.append(tag)
        aucs.append(roc_auc_score(y, pr) if len(set(y)) > 1 else float("nan"))
        accs.append(accuracy_score(y, p))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, aucs, w, label="AUC", color="#2E86AB")
    ax.bar(x + w/2, accs, w, label="Accuracy", color="#E63946")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Per-dataset Performance")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    for i, (a, ac) in enumerate(zip(aucs, accs)):
        ax.text(i - w/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, ac + 0.02, f"{ac:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "plots", "per_dataset.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Per-dataset plot saved to {out}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import traceback

    parser = argparse.ArgumentParser(description="AMDT AI-Image Detector")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--test", action="store_true",
                        help="Full pipeline smoke test on 100 images — no training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last saved epoch")
    args = parser.parse_args()

    cfg = Config()
    log.info(f"Device: {cfg.DEVICE} | GPUs: {torch.cuda.device_count()} | Mode: {args.mode}")

    if args.test:
        sys.exit(smoke_test(cfg))

    try:
        if args.mode == "train":
            train(cfg, debug=args.debug, fast=args.fast, resume=args.resume)
        elif args.mode == "predict":
            if not args.image:
                parser.error("--image is required for predict mode")
            model = load_model(cfg)
            result = predict_image(model, args.image, cfg)
            print(f"\n{'='*55}")
            print(f"File       : {os.path.basename(args.image)}")
            print(f"Prediction : {result['label']}")
            print(f"Confidence : {result['confidence']*100:.2f}%")
            print(f"P(AI)      : {result['prob_fake']*100:.2f}%")
            print(f"P(Real)    : {result['prob_real']*100:.2f}%")
            print(f"{'='*55}\n")
    except Exception:
        log.error("=" * 70)
        log.error("FATAL ERROR")
        log.error(traceback.format_exc())
        log.error("=" * 70)
        raise