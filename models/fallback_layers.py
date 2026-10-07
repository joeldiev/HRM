"""
CPU-compatible attention fallback for HRM layers.
Provides CPU/MPS compatible implementations when Flash Attention is not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """CPU/MPS compatible scaled dot-product attention."""
    
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # Apply causal mask if needed
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, value)
    
    return output


# Fallback flash attention function
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor, 
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False
) -> torch.Tensor:
    """Fallback flash attention implementation using standard PyTorch operations."""
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    
    # Reshape to standard attention format
    # Input: (batch, seqlen, nheads, headdim)
    # Output: (batch, nheads, seqlen, headdim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Apply scaling
    q = q * softmax_scale
    
    # Compute attention
    output = scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=causal
    )
    
    # Reshape back to flash attention format
    output = output.transpose(1, 2)
    
    if return_attn_probs:
        # Would need to compute attention probabilities
        # For now, just return output and None
        return output, None
    
    return output


@dataclass
class CosSin:
    """Rotary embedding cos/sin values."""
    cos: torch.Tensor
    sin: torch.Tensor


def rms_norm(x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMS Layer Normalization."""
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + variance_epsilon)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int) -> CosSin:
        """Generate cos/sin embeddings for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return CosSin(cos=cos, sin=sin)


def apply_rotary_emb(x: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
    """Apply rotary embedding to input tensor."""
    cos, sin = cos_sin.cos, cos_sin.sin
    
    # Split x into x1 and x2
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, hidden_size: int, expansion: float = 2.0):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    """Multi-head attention with rotary embeddings."""
    
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos_sin)
        k = apply_rotary_emb(k, cos_sin)
        
        # Use fallback attention
        output = flash_attn_func(q, k, v, causal=self.causal)
        
        # Reshape and project output
        output = output.contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


class CastedEmbedding(nn.Module):
    """Embedding layer with dtype casting."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.dtype = dtype
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids).to(self.dtype)


class CastedLinear(nn.Module):
    """Linear layer with dtype casting."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dtype = dtype
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.to(self.dtype)).to(x.dtype)