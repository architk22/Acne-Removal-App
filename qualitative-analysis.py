# Generated from: qualitative-analysis.ipynb
# Converted at: 2026-04-16T06:25:37.494Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================
# LIBRARIES
# ============================
import os
import glob
import random
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================
# CONFIG
# ============================
IMAGE_DIR  = "/kaggle/input/datasets/shreyash1110/acne04-yolov8/Images"
MODEL_PATH = "/kaggle/input/datasets/rhutpatel/trained-model-transunet/best_model.pth"

IMG_SIZE = 224
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT = "healthy clear skin, smooth natural skin texture, photorealistic"
NEGATIVE_PROMPT = "acne, blemish, scar, unrealistic, fake"

INPAINT_SIZE = 512

# ============================
# BASELINE (SWEET SPOT)
# ============================
BASE = {
    "guidance": 7.5,
    "steps": 30,
    "threshold": 0.5,
    "kernel": 9,
    "iter": 2,
    "prompt_strength": 1.0
}

# ============================
# PARAM SWEEPS
# ============================
param_sweeps = {
    "guidance_scale": [3.5, 7.5, 12.0],
    "num_inference_steps": [20, 30, 40],
    "confidence_threshold": [0.3, 0.5, 0.7],
    "mask_dilation": [(5,1), (9,2), (13,3)],
    "prompt_strength": [0.5, 1.0, 1.5],
}

# ============================
# TRANSUNET ARCHITECTURE
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
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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
                 mlp_ratio=4.0, dropout=0.1, pretrained=False):
        super().__init__()
        base      = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        self.patch_embed = PatchEmbedding(2048, embed_dim)
        max_seq          = (img_size // 32) ** 2
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_seq, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(num_layers)]
        )
        self.norm       = nn.LayerNorm(embed_dim)
        self.trans_proj = nn.Conv2d(embed_dim, 512, kernel_size=1)

        self.dec4 = DecoderBlock(512, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock( 64, 64, 32)
        self.dec0 = DecoderBlock( 32, 0, 16)
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
            pos = self.pos_embed.transpose(1, 2).reshape(1, -1, sz, sz)
            pos = F.interpolate(pos, (ph, pw), mode='bilinear', align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embed

        tokens = self.norm(self.transformer(tokens + pos))
        feat   = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], ph, pw)
        feat   = self.trans_proj(feat)

        x = self.dec4(feat, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.dec0(x, None)
        return self.head(x)


# ============================
# LOAD TRANSUNET
# ============================
print("Loading TransUNet segmentation model")
seg_model = TransUNet(num_classes=2, img_size=IMG_SIZE, pretrained=False).to(DEVICE)
seg_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
seg_model.eval()
print("Loaded\n")

preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ============================
# MASK FUNCTION
# ============================
def get_mask(img_rgb, threshold, kernel_size, iterations):
    h, w = img_rgb.shape[:2]
    tensor = preprocess(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = seg_model(tensor)

    probs = torch.softmax(logits, dim=1)[0,1].cpu().numpy()
    mask  = (probs > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.dilate(mask, kernel, iterations=iterations)

    return Image.fromarray(mask*255).convert("RGB")

# ============================
# PROMPT SCALING
# ============================
def scale_prompt(prompt, strength):
    return (prompt + ", ") * int(max(1, strength*2))

# ============================
# LOAD DIFFUSION
# ============================
print("Loading Stable Diffusion")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

# ============================
# SAMPLE IMAGES
# ============================
img_path = os.path.join(IMAGE_DIR, "levle2_89.jpg")

if not os.path.exists(img_path):
    raise FileNotFoundError("levle2_89.jpg not found in dataset")

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# ============================
# VISUALIZATION
# ============================
def visualize(param_name, values, img, mask, outputs):

    fig = plt.figure(figsize=(16, 4), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(1, len(values)+2, wspace=0.05)

    titles = ["Original", "Mask"] + [str(v) for v in values]

    for i, t in enumerate(titles):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(t, color="white", fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(gs[0,0])
    ax.imshow(img); ax.axis("off")

    ax = fig.add_subplot(gs[0,1])
    ax.imshow(mask); ax.axis("off")

    for i, out in enumerate(outputs):
        ax = fig.add_subplot(gs[0, i+2])
        ax.imshow(out); ax.axis("off")

    plt.suptitle(f"{param_name} Analysis", color="white")
    filename = f"{param_name.replace(' ', '_').lower()}_analysis.png"
    save_dir = '/kaggle/working/'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()

# ============================
# RUN ANALYSIS
# ============================

for param, values in param_sweeps.items():
    print(f"\nRunning {param} analysis")
    outputs = []

    for val in values:
        cfg = copy.deepcopy(BASE)

        # update one parameter
        if param == "guidance_scale":
            cfg["guidance"] = val
        elif param == "num_inference_steps":
            cfg["steps"] = val
        elif param == "confidence_threshold":
            cfg["threshold"] = val
        elif param == "mask_dilation":
            cfg["kernel"], cfg["iter"] = val
        elif param == "prompt_strength":
            cfg["prompt_strength"] = val

        # mask
        mask = get_mask(img,
                        cfg["threshold"],
                        cfg["kernel"],
                        cfg["iter"])

        # prompt
        prompt_scaled = scale_prompt(PROMPT, cfg["prompt_strength"])

        # diffusion
        img_pil = Image.fromarray(img).resize((512,512))
        mask_pil = mask.resize((512,512))

        out = pipe(
            prompt=prompt_scaled,
            negative_prompt=NEGATIVE_PROMPT,
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"]
        ).images[0]

        out = out.resize((img.shape[1], img.shape[0]))
        outputs.append(np.array(out))

    visualize(param, values, img, mask, outputs)

# ============================
# KERNEL SIZE ANALYSIS
# ============================

print("\nRunning kernel size analysis...")

kernel_sizes = [3, 5, 9, 13, 17]
outputs = []

for k in kernel_sizes:

    cfg = copy.deepcopy(BASE)
    cfg["kernel"] = k
    cfg["iter"]   = 2

    # mask
    mask = get_mask(img,
                    cfg["threshold"],
                    cfg["kernel"],
                    cfg["iter"])

    # diffusion
    img_pil = Image.fromarray(img).resize((512,512))
    mask_pil = mask.resize((512,512))

    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img_pil,
        mask_image=mask_pil,
        num_inference_steps=cfg["steps"],
        guidance_scale=cfg["guidance"]
    ).images[0]

    out = out.resize((img.shape[1], img.shape[0]))
    outputs.append(np.array(out))

# visualize
visualize("Kernel Size", kernel_sizes, img, mask, outputs)