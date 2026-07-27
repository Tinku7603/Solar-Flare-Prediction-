"""
model.py
--------
Temporal Convolutional Network (TCN) with an attention-pooling head for
SWAN-SF flare classification, extended to MULTI-TASK LEARNING:

  Task 1 (primary):   binary flare classification, as before.
  Task 2 (auxiliary):  self-supervised reconstruction of the input MVTS
                        sequence from the SAME pooled representation used
                        for classification.

Why this auxiliary task: the attention-pooled vector is a serious
bottleneck (e.g. 64 numbers reconstructing a 60-timestep x N-feature
sequence). Forcing that bottleneck to retain enough information to
rebuild the original series regularizes the shared TCN backbone toward
general, information-rich temporal representations, rather than
representations that only capture whatever narrow signal happens to
separate the (very rare) positive class in the current training fold.
This is a standard multi-task / self-supervised auxiliary-loss strategy
in representation learning, and — unlike adding a flare-intensity
auxiliary head — it needs no extra labels, so it is completely
unaffected by SMOTE/undersampling bookkeeping in sampling.py.
"""

import torch
import torch.nn as nn
from config import TCN_CHANNELS, KERNEL_SIZE, DROPOUT, N_CLASSES


class TemporalBlock(nn.Module):
    """One dilated causal-convolution residual block."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=DROPOUT):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.padding = padding

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(self.bn2(out))
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AttentionPool(nn.Module):
    """Learns a scalar attention weight per timestep, then weighted-sums over time."""

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


class ReconstructionDecoder(nn.Module):
    """
    Auxiliary head: maps the pooled representation back to the full
    (n_timesteps, n_features) input sequence. A deliberately narrow
    bottleneck -> wide reconstruction target, so the model must retain
    genuinely informative temporal structure to succeed at this task.
    """

    def __init__(self, pooled_dim, n_timesteps, n_features, hidden_mult=4):
        super().__init__()
        hidden = pooled_dim * hidden_mult
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_timesteps * n_features),
        )

    def forward(self, pooled):
        out = self.net(pooled)
        return out.view(-1, self.n_timesteps, self.n_features)


class TCNAttentionModel(nn.Module):
    def __init__(self, n_features, n_timesteps, n_classes=N_CLASSES, channels=None,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        channels = channels or TCN_CHANNELS

        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                         dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.attention = AttentionPool(channels[-1])
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, n_classes),
        )
        # Multi-task auxiliary head (self-supervised reconstruction)
        self.decoder = ReconstructionDecoder(channels[-1], n_timesteps, n_features)

    def forward(self, x):
        # x: (batch, timesteps, features) -> conv expects (batch, features, timesteps)
        x_in = x
        x = x.permute(0, 2, 1)
        feats = self.tcn(x)
        feats = feats.permute(0, 2, 1)
        pooled, attn_weights = self.attention(feats)
        logits = self.classifier(pooled)
        reconstruction = self.decoder(pooled)
        return logits, attn_weights, reconstruction


def build_model(n_features, n_timesteps, device):
    model = TCNAttentionModel(n_features=n_features, n_timesteps=n_timesteps)
    return model.to(device)
