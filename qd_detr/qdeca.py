"""
Query Decomposition and Event-Centric Attention (QDECA) Module

This module decomposes text queries into event, object, and temporal components,
then applies specialized cross-attention between video clips and each component.
The outputs are combined using learned clip-adaptive gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDecomposer(nn.Module):
    """
    Learns to decompose text queries into semantic components using attention.
    Uses learnable query vectors to extract event, object, and temporal information.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        # Learnable queries for each semantic type
        self.event_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.object_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.decompose_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=False)

        # Initialize queries with orthogonality for diversity
        nn.init.orthogonal_(self.event_query)
        nn.init.orthogonal_(self.object_query)
        nn.init.orthogonal_(self.temporal_query)

    def forward(self, src_txt, src_txt_mask):
        """
        Args:
            src_txt: (B, L_txt, D) text token features
            src_txt_mask: (B, L_txt) boolean mask (True = valid token)

        Returns:
            event_tokens: (B, L_txt, D) masked event features
            object_tokens: (B, L_txt, D) masked object features
            temporal_tokens: (B, L_txt, D) masked temporal features
            event_mask: (B, L_txt) boolean mask for events
            object_mask: (B, L_txt) boolean mask for objects
            temporal_mask: (B, L_txt) boolean mask for temporal
        """
        B, L_txt, D = src_txt.shape

        # Expand queries for batch
        queries = torch.cat([
            self.event_query.expand(B, -1, -1),
            self.object_query.expand(B, -1, -1),
            self.temporal_query.expand(B, -1, -1)
        ], dim=1)  # (B, 3, D)

        # Attention: queries attend to text tokens
        # PyTorch MultiheadAttention expects (seq_len, batch, dim) format
        _, attn_weights = self.decompose_attn(
            queries.transpose(0, 1),           # (3, B, D)
            src_txt.transpose(0, 1),           # (L_txt, B, D)
            src_txt.transpose(0, 1),           # (L_txt, B, D)
            key_padding_mask=~src_txt_mask,    # (B, L_txt) - True = ignore
            need_weights=True,
            average_attn_weights=True
        )
        # attn_weights: (B, 3, L_txt)

        # Create soft masks for each component
        # Keep all tokens but weight them by attention scores
        event_weights = attn_weights[:, 0, :].unsqueeze(-1)      # (B, L_txt, 1)
        object_weights = attn_weights[:, 1, :].unsqueeze(-1)     # (B, L_txt, 1)
        temporal_weights = attn_weights[:, 2, :].unsqueeze(-1)   # (B, L_txt, 1)

        # Apply soft masking
        event_tokens = src_txt * event_weights       # (B, L_txt, D)
        object_tokens = src_txt * object_weights     # (B, L_txt, D)
        temporal_tokens = src_txt * temporal_weights # (B, L_txt, D)

        # Create boolean masks for attention (use original text mask as base)
        event_mask = src_txt_mask.clone()
        object_mask = src_txt_mask.clone()
        temporal_mask = src_txt_mask.clone()

        return event_tokens, object_tokens, temporal_tokens, \
               event_mask, object_mask, temporal_mask


class QDECA(nn.Module):
    """
    Query Decomposition and Event-Centric Attention Module.

    Decomposes text queries into event, object, and temporal components,
    then applies separate cross-attention between video clips and each component.
    Results are combined using learned clip-adaptive gating.
    """

    def __init__(self, d_model, num_heads, max_txt_len=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Text decomposer
        self.decomposer = LearnedDecomposer(d_model, num_heads)

        # Separate cross-attention for each component
        self.event_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=False)
        self.object_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=False)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=False)

        # Clip-adaptive gating network
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Learnable null token for temporal component when no temporal words exist
        self.null_temporal_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Initialize gating MLP with small weights to avoid initial imbalance
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, src_vid, src_txt, src_txt_mask):
        """
        Args:
            src_vid: (B, L_vid, D) video clip features
            src_txt: (B, L_txt, D) text token features
            src_txt_mask: (B, L_txt) boolean mask (True = valid token)

        Returns:
            src_vid_qdeca: (B, L_vid, D) enhanced video features
        """
        B, L_vid, D = src_vid.shape

        # Decompose text into semantic components
        event_tokens, object_tokens, temporal_tokens, \
        event_mask, object_mask, temporal_mask = self.decomposer(src_txt, src_txt_mask)

        # Handle case where no temporal information exists
        # Check if all temporal weights are very small
        temporal_has_content = (temporal_tokens.abs().sum() > 1e-6)
        if not temporal_has_content:
            temporal_tokens = self.null_temporal_token.expand(B, 1, D)
            temporal_mask = torch.ones(B, 1, device=temporal_tokens.device, dtype=torch.bool)

        # Transpose for PyTorch MultiheadAttention (seq_len, batch, dim)
        vid_T = src_vid.transpose(0, 1)  # (L_vid, B, D)

        # Cross-attention: video clips attend to decomposed text components
        attn_e, _ = self.event_attn(
            vid_T,
            event_tokens.transpose(0, 1),
            event_tokens.transpose(0, 1),
            key_padding_mask=~event_mask  # True = ignore
        )  # (L_vid, B, D)

        attn_o, _ = self.object_attn(
            vid_T,
            object_tokens.transpose(0, 1),
            object_tokens.transpose(0, 1),
            key_padding_mask=~object_mask
        )  # (L_vid, B, D)

        attn_t, _ = self.temporal_attn(
            vid_T,
            temporal_tokens.transpose(0, 1),
            temporal_tokens.transpose(0, 1),
            key_padding_mask=~temporal_mask
        )  # (L_vid, B, D)

        # Transpose back to (B, L_vid, D)
        attn_e = attn_e.transpose(0, 1)
        attn_o = attn_o.transpose(0, 1)
        attn_t = attn_t.transpose(0, 1)

        # Clip-adaptive gating
        gate_logits = self.gate_mlp(src_vid)  # (B, L_vid, 3)
        g = F.softmax(gate_logits, dim=-1)    # (B, L_vid, 3)

        # Weighted combination of branch outputs
        src_vid_qdeca = (
            g[..., 0:1] * attn_e +
            g[..., 1:2] * attn_o +
            g[..., 2:3] * attn_t
        )  # (B, L_vid, D)

        return src_vid_qdeca

    def get_gate_weights(self, src_vid):
        """
        Utility function to extract gating weights for analysis.

        Args:
            src_vid: (B, L_vid, D) video clip features

        Returns:
            gate_weights: (B, L_vid, 3) softmax weights [event, object, temporal]
        """
        gate_logits = self.gate_mlp(src_vid)
        return F.softmax(gate_logits, dim=-1)
