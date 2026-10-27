import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 3D Convolution Block
# ==========================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ==========================================
# Transformer Block
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads=4,
        mlp_ratio=4
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim * mlp_ratio
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim * mlp_ratio,
                embed_dim
            )
        )

    def forward(self, x):

        # x shape:
        # Batch, Tokens, Features

        residual = x

        x = self.norm1(x)

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        x = residual + attention_output

        residual = x

        x = self.norm2(x)

        x = self.mlp(x)

        x = residual + x

        return x


# ==========================================
# Transformer Bottleneck
# ==========================================

class TransformerBottleneck(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=4,
        num_layers=2
    ):

        super().__init__()

        self.transformer_layers = nn.ModuleList([

            TransformerBlock(
                embed_dim=channels,
                num_heads=num_heads
            )

            for _ in range(num_layers)

        ])

    def forward(self, x):

        batch_size, channels, depth, height, width = x.shape

        # Convert 3D feature map into tokens
        x = x.view(
            batch_size,
            channels,
            depth * height * width
        )

        x = x.permute(
            0,
            2,
            1
        )

        # Transformer
        for layer in self.transformer_layers:

            x = layer(x)

        # Convert tokens back to 3D feature map
        x = x.permute(
            0,
            2,
            1
        )

        x = x.view(
            batch_size,
            channels,
            depth,
            height,
            width
        )

        return x


# ==========================================
# Multi Scale Fusion
# ==========================================

class MultiScaleFusion(nn.Module):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Conv3d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.BatchNorm3d(channels),

            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        feature1,
        feature2,
        feature3
    ):

        target_size = feature1.shape[2:]

        feature2 = F.interpolate(
            feature2,
            size=target_size,
            mode="trilinear",
            align_corners=False
        )

        feature3 = F.interpolate(
            feature3,
            size=target_size,
            mode="trilinear",
            align_corners=False
        )

        # Reduce feature dimensions if needed
        feature2 = feature2[:, :feature1.shape[1]]
        feature3 = feature3[:, :feature1.shape[1]]

        fused = feature1 + feature2 + feature3

        fused = self.fusion(fused)

        return fused


# ==========================================
# Decoder Block
# ==========================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose3d(

            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(
            out_channels * 2,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        # Handle possible size mismatch
        if x.shape[2:] != skip.shape[2:]:

            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x


# ==========================================
# Full Hybrid Model
# ==========================================

class HybridUNetTransformer(nn.Module):

    def __init__(
        self,
        in_channels=4,
        num_classes=4
    ):

        super().__init__()

        # Encoder
        self.encoder1 = ConvBlock(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = ConvBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool3d(2)

        self.encoder3 = ConvBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            128,
            256
        )

        self.transformer = TransformerBottleneck(
            channels=256,
            num_heads=8,
            num_layers=2
        )

        # Multi-scale projection
        self.proj1 = nn.Conv3d(
            32,
            128,
            kernel_size=1
        )

        self.proj2 = nn.Conv3d(
            64,
            128,
            kernel_size=1
        )

        self.proj3 = nn.Conv3d(
            128,
            128,
            kernel_size=1
        )

        self.multi_scale_fusion = MultiScaleFusion(
            channels=128
        )

        # Decoder
        self.decoder3 = DecoderBlock(
            256,
            128
        )

        self.decoder2 = DecoderBlock(
            128,
            64
        )

        self.decoder1 = DecoderBlock(
            64,
            32
        )

        # Final output
        self.output = nn.Conv3d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool3(e3)
        )

        # Transformer
        b = self.transformer(b)

        # Multi-scale feature fusion
        f1 = self.proj1(e1)
        f2 = self.proj2(e2)
        f3 = self.proj3(e3)

        fused = self.multi_scale_fusion(
            f3,
            f2,
            f1
        )

        # Decoder
        d3 = self.decoder3(
            b,
            fused
        )

        d2 = self.decoder2(
            d3,
            e2
        )

        d1 = self.decoder1(
            d2,
            e1
        )

        output = self.output(d1)

        return output