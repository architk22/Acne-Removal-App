import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


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
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
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
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self,
                 num_classes=2,
                 img_size=224,
                 embed_dim=768,
                 num_heads=12,
                 num_layers=12,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 pretrained=False):

        super().__init__()

        base = resnet50(
            weights=ResNet50_Weights.DEFAULT if pretrained else None
        )

        # Encoder
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool = base.maxpool
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4

        # Transformer
        self.patch_embed = PatchEmbedding(2048, embed_dim)

        max_seq = (img_size // 32) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.trans_proj = nn.Conv2d(embed_dim, 512, kernel_size=1)

        # Decoder
        self.dec4 = DecoderBlock(512, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)

        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Transformer
        tokens, ph, pw = self.patch_embed(s4)

        if tokens.shape[1] != self.pos_embed.shape[1]:
            sz = int(self.pos_embed.shape[1] ** 0.5)
            pos = self.pos_embed.transpose(1, 2).reshape(1, -1, sz, sz)
            pos = F.interpolate(pos, (ph, pw), mode='bilinear', align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embed

        tokens = self.norm(self.transformer(tokens + pos))

        feat = tokens.transpose(1, 2).reshape(-1, tokens.shape[-1], ph, pw)
        feat = self.trans_proj(feat)

        # Decoder
        x = self.dec4(feat, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = self.dec1(x, s0)
        x = self.dec0(x, None)

        return self.head(x)