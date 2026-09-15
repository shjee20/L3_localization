
import torch.nn as nn
import torchvision.models as models
import torch


class ResNet34BinaryCT(nn.Module):

    def __init__(self, pretrained=True, init_conv1_from_rgb=True):
        super().__init__()
        self.backbone = models.resnet34(pretrained=pretrained)

        # --- conv1 1채널로 교체 ---
        old_conv1 = self.backbone.conv1  # (64,3,7,7) if pretrained

        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )

        if pretrained and init_conv1_from_rgb and old_conv1.weight.shape[1] == 3:
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

        self.backbone.conv1 = new_conv1

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):  # x: (B,1,H,W)
        return self.backbone(x)  # logits: (B,1)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))  # probs: (B,1)


def _make_resnet34_grayscale_encoder(pretrained=True, init_conv1_from_rgb=True):
    backbone = models.resnet34(pretrained=pretrained)

    old_conv1 = backbone.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )

    if pretrained and init_conv1_from_rgb and old_conv1.weight.shape[1] == 3:
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

    backbone.conv1 = new_conv1
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class ContextResNet34ManyToOne(nn.Module):
    """
    Context-aware many-to-one axial model.

    Input:
        x: (B, T, 1, H, W)
    Output:
        logits for the center slice only: (B, 1)
    """

    def __init__(
        self,
        pretrained=True,
        init_conv1_from_rgb=True,
        context_dropout=0.2,
    ):
        super().__init__()
        self.encoder, feature_dim = _make_resnet34_grayscale_encoder(
            pretrained=pretrained,
            init_conv1_from_rgb=init_conv1_from_rgb,
        )

        self.context_1dcnn = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(context_dropout),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (B,T,1,H,W), got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        if c != 1:
            raise ValueError(f"Expected single-channel CT slices, got C={c}")
        if t % 2 == 0:
            raise ValueError(f"seq_len T must be odd for center-slice prediction, got T={t}")

        x = x.reshape(b * t, c, h, w)
        feat = self.encoder(x)              # (B*T, 512)
        feat = feat.reshape(b, t, -1)       # (B, T, 512)
        feat = feat.transpose(1, 2)         # (B, 512, T)
        ctx = self.context_1dcnn(feat)      # (B, 128, T)
        ctx = ctx.transpose(1, 2)           # (B, T, 128)
        center_feat = ctx[:, t // 2, :]     # (B, 128)
        return self.fc(center_feat)         # (B, 1)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class ContextResNet34ManyToManyTransformer(nn.Module):
    """
    Position-aware many-to-many axial sequence model.

    Input:
        x: (B, T, 1, H, W)
    Output:
        slice-wise logits for the same local window: (B, T)
    """

    def __init__(
        self,
        seq_len=5,
        pretrained=True,
        init_conv1_from_rgb=True,
        d_model=256,
        num_transformer_layers=1,
        nhead=4,
        dim_feedforward=512,
        transformer_dropout=0.1,
        attn_dist_alpha=0.0,
        attn_dist_mode="none",
    ):
        super().__init__()
        if seq_len < 1 or seq_len % 2 == 0:
            raise ValueError(f"seq_len must be a positive odd integer, got {seq_len}")

        self.seq_len = seq_len
        if attn_dist_mode not in {"none", "gaussian", "laplace"}:
            raise ValueError(
                f"attn_dist_mode must be one of ['none', 'gaussian', 'laplace'], got {attn_dist_mode!r}"
            )
        self.attn_dist_alpha = float(attn_dist_alpha)
        self.attn_dist_mode = attn_dist_mode

        self.encoder, feature_dim = _make_resnet34_grayscale_encoder(
            pretrained=pretrained,
            init_conv1_from_rgb=init_conv1_from_rgb,
        )
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.rel_pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        if self.attn_dist_mode == "none" or self.attn_dist_alpha <= 0:
            self.register_buffer("attn_dist_bias", None, persistent=False)
        else:
            positions = torch.arange(seq_len, dtype=torch.float32)
            dist = (positions[:, None] - positions[None, :]).abs()
            if self.attn_dist_mode == "gaussian":
                bias = -self.attn_dist_alpha * dist.pow(2)
            else:
                bias = -self.attn_dist_alpha * dist
            self.register_buffer("attn_dist_bias", bias, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )
        self.token_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.rel_pos_embed, std=0.02)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (B,T,1,H,W), got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        if t != self.seq_len:
            raise ValueError(f"Expected seq_len T={self.seq_len}, got T={t}")
        if c != 1:
            raise ValueError(f"Expected single-channel CT slices, got C={c}")

        x = x.reshape(b * t, c, h, w)
        feat = self.encoder(x)                  # (B*T, 512)
        feat = feat.reshape(b, t, -1)           # (B, T, 512)
        tokens = self.feature_proj(feat)        # (B, T, d_model)
        tokens = tokens + self.rel_pos_embed    # preserve relative slice position
        attn_mask = None
        if self.attn_dist_bias is not None:
            attn_mask = self.attn_dist_bias.to(device=tokens.device, dtype=tokens.dtype)
        ctx = self.transformer_encoder(tokens, mask=attn_mask)  # (B, T, d_model)
        return self.token_head(ctx).squeeze(-1) # (B, T)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
