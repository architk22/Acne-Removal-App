# Generated from: diffusion-inpaint.ipynb
# Converted at: 2026-04-14T16:07:12.807Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ============================
# LOAD LIBRARIES
# ============================
import os
import glob
import random
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
IMAGE_DIR      = "/kaggle/input/datasets/shreyash1110/acne04-yolov8/Images"
MODEL_PATH     = "/kaggle/input/datasets/rhutpatel/trained-model-transunet/best_model.pth"
SAVE_DIR       = "/kaggle/working/"

IMG_SIZE       = 224
NUM_SAMPLES    = 4
MASK_THRESHOLD = 0.5        
DILATION_KERNEL= 9
DILATION_ITER  = 2

SD_MODEL_ID    = "runwayml/stable-diffusion-inpainting"
INPAINT_STEPS  = 30
GUIDANCE_SCALE = 7.5
INPAINT_SIZE   = 512

# Prompt: tell diffusion model what to fill in
PROMPT = ("healthy clear skin, smooth natural skin texture,"
    "photorealistic, high quality portrait")

NEGATIVE_PROMPT = ("acne, pimple, blemish, redness, spot, lesion, scar,"
    "blurry, fake, painted, airbrushed, overexposed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

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

# Preprocessing (no augmentation - same as val/test)
preprocess = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ============================
# MASK GENERATION: IMAGE -> TransUNet -> DILATED BINARY MASK
# ============================
def get_inpaint_mask(img_rgb: np.ndarray) -> Image.Image:
    orig_h, orig_w = img_rgb.shape[:2]

    transformed = preprocess(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = seg_model(tensor)
    probs  = torch.softmax(logits, dim=1)[0, 1]
    mask   = (probs > MASK_THRESHOLD).cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATION_KERNEL, DILATION_KERNEL)
    )
    mask = cv2.dilate(mask, kernel, iterations=DILATION_ITER)

    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    mask_pil = mask_pil.resize((INPAINT_SIZE, INPAINT_SIZE), Image.NEAREST)
    return mask_pil


# ============================
# LOAD STABLE DIFFUSION INPAINTING PIPELINE
# ============================
print("\nLoading Stable Diffusion Inpainting Model")
print(f"Model: {SD_MODEL_ID}")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    SD_MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = pipe.to(DEVICE)

if DEVICE == "cuda":
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


# ============================
# INPAINTING FUNCTION
# ============================
def inpaint_image(img_rgb: np.ndarray, mask_pil: Image.Image) -> np.ndarray:
    orig_h, orig_w = img_rgb.shape[:2]

    img_pil = Image.fromarray(img_rgb).resize(
        (INPAINT_SIZE, INPAINT_SIZE), Image.LANCZOS
    )

    result = pipe(
        prompt          = PROMPT,
        negative_prompt = NEGATIVE_PROMPT,
        image           = img_pil,
        mask_image      = mask_pil,
        num_inference_steps = INPAINT_STEPS,
        guidance_scale  = GUIDANCE_SCALE,
    ).images[0]

    result = result.resize((orig_w, orig_h), Image.LANCZOS)
    return np.array(result)


# ============================
# SAMPLE IMAGES
# ============================
all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")))
all_images = [p for p in all_images
              if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Pick images that have acne
# Note: Label is not used at all in cleaning acne
LABEL_DIR = "/kaggle/input/datasets/shreyash1110/acne04-yolov8/labels/content/labels"
has_labels = []
for p in all_images:
    base = os.path.splitext(os.path.basename(p))[0]
    lbl  = os.path.join(LABEL_DIR, f"{base}.txt")
    if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
        has_labels.append(p)

random.seed(42)
samples = random.sample(has_labels, min(NUM_SAMPLES, len(has_labels)))


# ============================
# MAIN LOOP
# ============================
results = []

for i, img_path in enumerate(samples):
    fname = os.path.basename(img_path)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("\n1. Generating segmentation mask")
    mask_pil = get_inpaint_mask(img_rgb)
    mask_arr = np.array(mask_pil.convert("L"))
    lesion_pixels = (mask_arr > 127).sum()
    coverage = lesion_pixels / mask_arr.size * 100
    print(f"    Mask coverage: {coverage:.1f}% of image")

    if lesion_pixels == 0:
        print("2. No lesions detected - use original")
        inpainted = img_rgb.copy()
    else:
        print("2. Running Stable Diffusion Inpainting")
        inpainted = inpaint_image(img_rgb, mask_pil)
        print("3. Done")

    mask_np = np.array(mask_pil.convert("L").resize(
        (img_rgb.shape[1], img_rgb.shape[0]), Image.NEAREST
    ))
    results.append((img_rgb, mask_np, inpainted, fname))
    out_path = os.path.join(SAVE_DIR, f"restored_{fname}")
    cv2.imwrite(out_path, cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))

# ============================
# VISUALISATION
# ============================
print("Generating visualisation grid")
n = len(results)
fig = plt.figure(figsize=(15, 5 * n), facecolor="#0d0d0d")
gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.08, wspace=0.04, left=0.02, right=0.98, top=0.90, bottom=0.02)

col_titles = ["Original\n(Before)", "Acne Mask\n(TransUNet)", "Restored\n(After Inpainting)"]
col_colors = ["#e0e0e0", "#ff9966", "#66ffb2"]

for col_idx, (title, color) in enumerate(zip(col_titles, col_colors)):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

for row, (img_rgb, mask_np, inpainted, fname) in enumerate(results):
    # Original
    ax0 = fig.add_subplot(gs[row, 0])
    ax0.imshow(img_rgb)
    ax0.set_ylabel(fname, color="#aaaaaa", fontsize=8, labelpad=4)
    ax0.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax0.spines.values():
        spine.set_edgecolor("#333333")

    # Mask overlay
    ax1 = fig.add_subplot(gs[row, 1])
    overlay = img_rgb.copy()
    lesion_region = mask_np > 127
    overlay[lesion_region] = (
        overlay[lesion_region] * 0.35
        + np.array([255, 80, 40]) * 0.65
    ).clip(0, 255).astype(np.uint8)
    ax1.imshow(overlay)
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    # Inpainted
    ax2 = fig.add_subplot(gs[row, 2])
    ax2.imshow(inpainted)
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

fig.suptitle("Acne Removal Pipeline: TransUNet Segmentation + Stable Diffusion Inpainting",
             color="white", fontsize=15, fontweight="bold", y=0.99)

out_grid = os.path.join(SAVE_DIR, "before_after_grid.png")
plt.savefig(out_grid, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()


# ============================
# DIFFERENCE MAP
# ============================
print("\nGenerating per-image difference maps")
fig2, axes = plt.subplots(n, 4, figsize=(18, 4.5 * n), facecolor="#0d0d0d")
if n == 1:
    axes = axes[np.newaxis, :]

col_labels = ["Before", "Mask", "After", "Difference"]
for col, label in enumerate(col_labels):
    axes[0, col].set_title(label, color="white", fontsize=12, pad=10)

for row, (img_rgb, mask_np, inpainted, fname) in enumerate(results):
    # Difference amplified ×3 for visibility
    diff = np.abs(img_rgb.astype(np.int32) - inpainted.astype(np.int32))
    diff_amp = np.clip(diff * 3, 0, 255).astype(np.uint8)

    for col, data in enumerate([img_rgb, None, inpainted, diff_amp]):
        ax = axes[row, col]
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if col == 1:
            ax.imshow(mask_np, cmap="hot", vmin=0, vmax=255)
        else:
            ax.imshow(data)

        if col == 0:
            ax.set_ylabel(fname, color="#aaaaaa", fontsize=7.5, labelpad=4)

fig2.suptitle("Difference Analysis: What the Diffusion Model Changed", color="white", fontsize=14, fontweight="bold", y=0.99)
fig2.tight_layout()

out_diff = os.path.join(SAVE_DIR, "difference_maps.png")
fig2.savefig(out_diff, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.show()