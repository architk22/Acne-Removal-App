# Generated from: train-transunet.ipynb
# Converted at: 2026-04-14T16:06:16.292Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================
# TRANSUNET TRAINING
# ============================
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================
# CONFIG
# ============================
IMAGE_DIR   = "/kaggle/input/datasets/shreyash1110/acne04-yolov8/Images"
LABEL_DIR   = "/kaggle/input/datasets/shreyash1110/acne04-yolov8/labels/content/labels"
SAVE_DIR    = "/kaggle/working/"

IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 100
LR          = 1e-4
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10
NUM_WORKERS = 2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# DATASET
# ============================
def bbox_to_mask(label_path, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            cx_px = int(cx * img_w)
            cy_px = int(cy * img_h)
            rx    = max(1, int((bw * img_w) / 2))
            ry    = max(1, int((bh * img_h) / 2))
            cv2.ellipse(mask, (cx_px, cy_px), (rx, ry), 0, 0, 360, 255, -1)
            
    return (mask > 0).astype(np.uint8)


def make_transforms(img_size, augment):
    if augment:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class AcneDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=224, augment=True):
        self.img_size = img_size
        self.tf       = make_transforms(img_size, augment)
        self.samples  = []

        for img_path in sorted(glob.glob(os.path.join(image_dir, "*.*"))):
            if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            base     = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(label_dir, f"{base}.txt")
            self.samples.append((img_path, lbl_path))

        print(f"Found {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        mask = bbox_to_mask(lbl_path, H, W)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


# ============================
# AUGMENTATION WRAPPER
# ============================
class _AugWrapper(Dataset):
    def __init__(self, subset, image_dir, label_dir, img_size):
        self.subset = subset
        self.tf     = make_transforms(img_size, augment=True)
        self.base   = subset.dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        real_idx = self.subset.indices[idx]
        img_path, lbl_path = self.base.samples[real_idx]
        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        mask = bbox_to_mask(lbl_path, H, W)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


def get_loaders(image_dir, label_dir, img_size, batch_size,
                val_split, test_split, num_workers):

    full = AcneDataset(image_dir, label_dir, img_size, augment=False)

    n    = len(full)
    n_te = int(n * test_split)
    n_va = int(n * val_split)
    n_tr = n - n_va - n_te

    train_ds, val_ds, test_ds = random_split(
        full,
        [n_tr, n_va, n_te],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = _AugWrapper(train_ds, image_dir, label_dir, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Split -> Train: {n_tr}  Val: {n_va}  Test: {n_te}")
    return train_loader, val_loader, test_loader


# ============================
# TRANSUNET MODEL
# ============================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, num_classes=2, img_size=224,
                 embed_dim=768, num_heads=12, num_layers=12,
                 mlp_ratio=4.0, dropout=0.1, pretrained=True):

        super().__init__()

        base = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.patch_embed = PatchEmbedding(2048, embed_dim)

        max_seq = (img_size // 32) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, embed_dim))

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(num_layers)]
        )

        self.norm       = nn.LayerNorm(embed_dim)
        self.trans_proj = nn.Conv2d(embed_dim, 512, kernel_size=1)

        self.dec4 = DecoderBlock(512, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)

        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        tokens, ph, pw = self.patch_embed(s4)

        if tokens.shape[1] != self.pos_embed.shape[1]:
            sz  = int(self.pos_embed.shape[1] ** 0.5)
            pos = self.pos_embed.transpose(1,2).reshape(1, -1, sz, sz)
            pos = F.interpolate(pos, (ph, pw), mode='bilinear', align_corners=False)
            pos = pos.flatten(2).transpose(1,2)
        else:
            pos = self.pos_embed

        tokens = self.norm(self.transformer(tokens + pos))

        feat = tokens.transpose(1,2).reshape(-1, tokens.shape[-1], ph, pw)
        feat = self.trans_proj(feat)

        x = self.dec4(feat, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.dec0(x, None)

        return self.head(x)


# ============================
# LOSS + DICE SCORE
# ============================
class SegLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.ce     = nn.CrossEntropyLoss()
        self.smooth = smooth

    def dice(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]
        tgt   = (targets == 1).float()
        inter = (probs * tgt).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        return (1 - (2 * inter + self.smooth) / (union + self.smooth)).mean()

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def dice_score(logits, targets, smooth=1e-5):
    preds   = (logits.argmax(dim=1) == 1).float()
    tgt     = (targets == 1).float()
    inter   = (preds * tgt).sum(dim=(1, 2))
    union   = preds.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()

# ============================
# Train
# ============================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = total_dice = n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits      = model(imgs)
            loss        = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(logits, masks)
            n          += 1

    return total_loss / n, total_dice / n


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data")
    train_loader, val_loader, _ = get_loaders(
        IMAGE_DIR, LABEL_DIR, IMG_SIZE, BATCH_SIZE,
        VAL_SPLIT, TEST_SPLIT, NUM_WORKERS
    )

    print("\nBuilding model")
    model = TransUNet(num_classes=2, img_size=IMG_SIZE, pretrained=True).to(DEVICE)
    criterion = SegLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_dice = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_dice = run_epoch(model, val_loader,   criterion, None)
        scheduler.step()

        print(f"\nEp {epoch:03d}/{EPOCHS}")
        print(f"Train loss {tr_loss:.4f}, dice {tr_dice:.4f}")
        print(f"Val loss {va_loss:.4f}, dice {va_dice:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"New best saved (val dice {best_dice:.4f})")

    print("Training complete")


if __name__ == "__main__":
    train()