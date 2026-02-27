"""
Transformer model implementation from scratch.
Implements all components needed for a decoder-only transformer language model.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =============================================================================
# Problem (linear): Implementing the linear module
# =============================================================================

class Linear(nn.Module):
    """
    Linear transformation layer: y = xW^T
    
    Note: We don't use bias in modern transformer implementations (like LLaMA).
    """
    
    def __init__(self, d_in: int, d_out: int):
        """
        Initialize linear layer.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Weight matrix of shape (d_out, d_in)
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply linear transformation: y = x @ W^T
        
        Args:
            x: Input tensor of shape (..., d_in)
        
        Returns:
            Output tensor of shape (..., d_out)
        """
        # Apply matrix multiplication using the transpose of the weight matrix
        # [Optimization]: F.linear uses optimized CUDA kernels
        return F.linear(x, self.weight)

# =============================================================================
# Problem (embedding): Implement the embedding module
# =============================================================================

class Embedding(nn.Module):
    """
    Token embedding layer that maps token indices to dense vectors.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Embedding weight matrix of shape (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings from normal distribution."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Tensor of token indices of shape (batch, seq_len)
        
        Returns:
            Tensor of embeddings of shape (batch, seq_len, d_model)
        """
        # Retrieve embeddings for the provided token indices
        return F.embedding(token_ids, self.weight)

# =============================================================================
# Problem (rmsnorm): Root Mean Square Layer Normalization
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is a simplification of LayerNorm that removes the mean centering
    and only normalizes by the root mean square of the activations.
    
    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize RMSNorm.
        
        Args:
            d_model: Model dimension (size of last dimension)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization.
        
        RMSNorm(x) = x / RMS(x) * gamma
        where RMS(x) = sqrt(mean(x^2) + eps)
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # [Optimization]: Keep computations in float32 for stability
        input_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Calculate the mean of squares for each feature vector
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # Compute the root mean square, adding epsilon for numerical stability
        # [Optimization]: rsqrt is slightly faster than 1/sqrt
        rms = torch.rsqrt(mean_square + self.eps)
        # Normalize the input and apply the learned scale parameter
        return (x * rms).to(input_dtype) * self.weight

# =============================================================================
# Problem (softmax): Implement softmax (used in attention)
# =============================================================================

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # [Optimization]: Use PyTorch's optimized implementation
    # This handles numerical stability (subtracting max) internally and efficiently
    return F.softmax(x, dim=dim)

# =============================================================================
# SiLU activation (helper for SwiGLU)
# =============================================================================

def silu(x: Tensor) -> Tensor:
    """
    SiLU (Sigmoid Linear Unit) activation function.
    https://arxiv.org/abs/1702.03118
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with SiLU applied element-wise
    """
    # [Optimization]: Use optimized C++ implementation
    return F.silu(x)

# =============================================================================
# Problem (positionwise_feedforward): Implement the position-wise feed-forward network
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    https://arxiv.org/pdf/2002.05202
    
    SwiGLU is a variant of the GLU (Gated Linear Unit) that uses SiLU activation.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize SwiGLU layer.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward layer
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gate projection: d_model -> d_ff
        self.w1 = Linear(d_model, d_ff)
        # Down projection: d_ff -> d_model
        self.w2 = Linear(d_ff, d_model)
        # Up projection: d_model -> d_ff
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        # Apply the SwiGLU operation: (SiLU(xW1) * xW3)W2
        return self.w2(silu(self.w1(x)) * self.w3(x))

# =============================================================================
# Problem (rope): Implement RoPE (Rotary Position Embedding)
# =============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE encodes position information by rotating the query and key vectors
    in a way that makes the dot product depend on relative position.
    """
    
    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            d_model: Model dimension (head dimension for attention)
            max_seq_len: Maximum sequence length
            theta: Base for frequency computation (default: 10000.0)
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        # inv_freq shape: (d_model // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values for positions up to seq_len."""
        # positions shape: (seq_len,)
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        
        # freqs shape: (seq_len, d_model // 2)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Duplicate each frequency for the pair of dimensions
        # emb shape: (seq_len, d_model)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotate half the hidden dims of the input.
        """
        # Split the last dimension into two halves and construct the rotated vector
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        Apply rotary position embedding.
        
        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, d_k)
               or (..., seq_len, d_model)
            token_positions: Position indices of shape (batch, seq_len) or (seq_len,)
        
        Returns:
            Tensor with rotary position embedding applied, same shape as input
        """
        # [Optimization]: Handle cases where input sequence is longer than cached
        seq_len = x.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
             self._precompute_cache(seq_len)

        # Look up precomputed cos/sin for the given positions
        # cos/sin shape: (batch, seq_len, d_model) (after indexing)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # Broadcast over heads dimension if the input is 4D (batch, heads, seq, d_k)
        if x.dim() == 4:
            # Unsqueeze head dimension: (batch, 1, seq, d_k)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        # Apply the rotation formula: x * cos + rotate_half(x) * sin
        return (x * cos) + (self._rotate_half(x) * sin)

def apply_rope(x: Tensor, d_model: int, theta: float, max_seq_len: int, token_positions: Tensor) -> Tensor:
    """
    Functional interface for applying RoPE.
    
    Args:
        x: Input tensor of shape (..., seq_len, d_model)
        d_model: Dimension of the model/head
        theta: RoPE base frequency
        max_seq_len: Maximum sequence length
        token_positions: Position indices
    
    Returns:
        Tensor with RoPE applied
    """
    rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
    rope = rope.to(x.device)
    return rope(x, token_positions)

# =============================================================================
# Problem (scaled_dot_product_attention): Implement scaled dot-product attention
# =============================================================================

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor of shape (..., seq_len_q, d_k)
        K: Key tensor of shape (..., seq_len_k, d_k)
        V: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask of shape (..., seq_len_q, seq_len_k)
              True values indicate positions to attend to, False positions are masked
    
    Returns:
        Attention output of shape (..., seq_len_q, d_v)
    """
    # [Optimization]: Use PyTorch's optimized F.scaled_dot_product_attention (SDPA).
    # This automatically uses Flash Attention or Memory-Efficient Attention on GPUs.
    # It is significantly faster and uses less memory than manual implementation.
    
    # Heuristic to detect if we can use the `is_causal=True` optimization:
    # If a mask is provided, and it's a square lower-triangular mask, we treat it as causal.
    is_causal = False
    if mask is not None:
        L_q, L_k = Q.size(-2), K.size(-2)
        if L_q == L_k and mask.shape[-2:] == (L_q, L_k):
            # If the mask is provided, we can potentially assume it's causal 
            # and let SDPA handle the masking logic efficiently.
            is_causal = True
            # When using is_causal=True, we should pass None as attn_mask to SDPA
            # to let it generate the optimized causal mask internally.
            mask = None 

    try:
        # F.scaled_dot_product_attention is available in torch >= 2.0
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=is_causal)
    except Exception:
        # Fallback to manual implementation if specific hardware/version doesn't support SDPA
        d_k = Q.shape[-1]
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # Shapes: (..., Lq, d) @ (..., d, Lk) -> (..., Lq, Lk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Masked fill: set masked positions (False) to -inf to ignore them in softmax
            scores = scores.masked_fill(mask == False, float('-inf'))
        elif is_causal:
            # Re-create mask if we cleared it for SDPA check but fell back here
            L = Q.size(-2)
            causal_mask = torch.tril(torch.ones(L, L, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(causal_mask == False, float('-inf'))
        
        # Apply softmax to obtain attention weights
        attn_weights = softmax(scores, dim=-1)
        
        # Compute weighted sum of values: weights @ V
        # Shapes: (..., Lq, Lk) @ (..., Lk, d_v) -> (..., Lq, d_v)
        return torch.matmul(attn_weights, V)

# =============================================================================
# Problem (multihead_self_attention): Implement causal multi-head self-attention
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer with causal masking.
    
    This implements the attention mechanism used in decoder-only transformers
    like GPT and LLaMA. It projects the input into queries, keys, and values,
    applies scaled dot-product attention with causal masking, and projects back.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) attention mask."""
        # mask[i, j] = True if j <= i (can attend to position j from position i)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to Query, Key, and Value representations
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to split heads and transpose to (batch, heads, seq, d_k)
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create causal mask to prevent attending to future tokens
        mask = self._create_causal_mask(seq_len, x.device)
        
        # Compute scaled dot-product attention
        # Output shape: (batch, heads, seq, d_k)
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # Concatenate heads and project output back to d_model size
        # (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.output_proj(attn_output)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embedding (RoPE).
    
    This extends the basic multi-head attention by applying RoPE to the
    query and key vectors before computing attention scores.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0):
        """
        Initialize multi-head self-attention with RoPE.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE base frequency
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # RoPE for query/key rotation
        self.rope = RotaryPositionEmbedding(self.d_k, max_seq_len, theta)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply multi-head self-attention with RoPE.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices of shape (batch, seq_len)
                           If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Default to sequential positions if none are provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Project inputs to Query, Key, and Value representations
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to split heads and transpose
        # (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE to Query and Key vectors
        # RoPE handles broadcasting for the head dimension internally
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len, x.device)
        
        # Compute scaled dot-product attention using rotated queries and keys
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.output_proj(attn_output)

# =============================================================================
# Problem (transformer_block): Implement the Transformer block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.
    
    Structure (Pre-LN / LLaMA-style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            theta: RoPE base frequency
            eps: Epsilon for layer normalization
        """
        super().__init__()
        
        # Layer norms (Pre-LN)
        self.ln1 = RMSNorm(d_model, eps)
        self.ln2 = RMSNorm(d_model, eps)
        
        # Self-attention with RoPE
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Apply Transformer block (Pre-LN style).

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional position indices
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Apply self-attention block with residual connection and normalization
        # x = x + Attention(Norm(x))
        residual = x
        x = self.ln1(x)
        x = self.attn(x, token_positions)
        x = residual + x
        
        # Apply feed-forward block with residual connection and normalization
        # x = x + FFN(Norm(x))
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

# =============================================================================
# Problem (transformer_lm): Implementing the Transformer LM
# =============================================================================

class TransformerLM(nn.Module):
    """
    Transformer Language Model (decoder-only, like GPT/LLaMA).
    
    Architecture:
        1. Token embedding
        2. N x Transformer blocks
        3. Final layer norm
        4. Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        """
        Initialize Transformer LM.
        
        Args:
            vocab_size: Size of vocabulary
            context_length: Maximum sequence/context length
            d_model: Model dimension
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            rope_theta: RoPE base frequency
            eps: Epsilon for layer normalization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, eps)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_ln = RMSNorm(d_model, eps)
        
        # Output projection (to vocab size)
        self.output = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Transformer LM.
        
        Args:
            token_ids: Input token indices of shape (batch, seq_len)
            token_positions: Optional position indices of shape (batch, seq_len)
                           If None, uses sequential positions [0, 1, 2, ...]
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Default to sequential positions if none are provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get token embeddings
        x = self.token_embeddings(token_ids)
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Apply final layer normalization
        x = self.final_ln(x)
        
        # Project to vocabulary size to get logits
        logits = self.output(x)
        
        return logits
    
    def load_weights(self, state_dict: dict):
        """
        Load weights from a state dict.
        
        Args:
            state_dict: Dictionary mapping weight names to tensors
        """
        # Token embeddings
        if "token_embeddings.weight" in state_dict:
            self.token_embeddings.weight.data.copy_(state_dict["token_embeddings.weight"])
        
        # Output projection
        if "output.weight" in state_dict:
            self.output.weight.data.copy_(state_dict["output.weight"])
        
        # Final layer norm
        if "final_ln.weight" in state_dict:
            self.final_ln.weight.data.copy_(state_dict["final_ln.weight"])
        
        # Layer weights
        for layer_idx, layer in enumerate(self.layers):
            prefix = f"layers.{layer_idx}"
            
            # Layer norms
            if f"{prefix}.ln1.weight" in state_dict:
                layer.ln1.weight.data.copy_(state_dict[f"{prefix}.ln1.weight"])
            if f"{prefix}.ln2.weight" in state_dict:
                layer.ln2.weight.data.copy_(state_dict[f"{prefix}.ln2.weight"])
            
            # Attention projections
            if f"{prefix}.attn.q_proj.weight" in state_dict:
                layer.attn.q_proj.weight.data.copy_(state_dict[f"{prefix}.attn.q_proj.weight"])
            if f"{prefix}.attn.k_proj.weight" in state_dict:
                layer.attn.k_proj.weight.data.copy_(state_dict[f"{prefix}.attn.k_proj.weight"])
            if f"{prefix}.attn.v_proj.weight" in state_dict:
                layer.attn.v_proj.weight.data.copy_(state_dict[f"{prefix}.attn.v_proj.weight"])
            if f"{prefix}.attn.output_proj.weight" in state_dict:
                layer.attn.output_proj.weight.data.copy_(state_dict[f"{prefix}.attn.output_proj.weight"])
            
            # FFN weights
            if f"{prefix}.ffn.w1.weight" in state_dict:
                layer.ffn.w1.weight.data.copy_(state_dict[f"{prefix}.ffn.w1.weight"])
            if f"{prefix}.ffn.w2.weight" in state_dict:
                layer.ffn.w2.weight.data.copy_(state_dict[f"{prefix}.ffn.w2.weight"])
            if f"{prefix}.ffn.w3.weight" in state_dict:
                layer.ffn.w3.weight.data.copy_(state_dict[f"{prefix}.ffn.w3.weight"])

# =============================================================================
# Problem (transformer_accounting): Transformer LM resource accounting
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())

def count_flops_per_token(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> int:
    """
    Estimate the number of FLOPs per token for a forward pass.
    
    This is an approximation that counts multiply-accumulate operations (MACs).
    Each MAC is typically counted as 2 FLOPs.
    
    Args:
        vocab_size: Size of vocabulary
        context_length: Maximum sequence length (used for attention)
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
    
    Returns:
        Approximate FLOPs per token
    """
    # FLOPs estimation per layer breakdown:
    # Attention mechanism:
    # Q, K, V projections: 3 * (d_model * d_model) MACs
    # Output projection: d_model * d_model MACs
    # Total projections: 4 * d_model^2 MACs = 8 * d_model^2 FLOPs
    # Attention scores (Q @ K^T): context_length * d_model MACs per head per token
    # Value aggregation (Scores @ V): context_length * d_model MACs per head per token
    # Total Attention ops per token: 4 * d_model * context_length FLOPs
    
    # FFN (SwiGLU):
    # 3 linear projections (w1, w2, w3): 3 * (d_model * d_ff) MACs = 6 * d_model * d_ff FLOPs
    
    # Logits calculation:
    # d_model * vocab_size MACs = 2 * d_model * vocab_size FLOPs
    
    flops_per_layer = (
        8 * d_model**2 +                  # Attention linear projections
        4 * d_model * context_length +    # Attention dot products
        6 * d_model * d_ff                # FFN linear projections
    )
    
    total_flops = num_layers * flops_per_layer + 2 * d_model * vocab_size
    
    return total_flops

def estimate_memory_bytes(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    dtype_bytes: int = 4,  # float32 = 4 bytes
) -> int:
    """
    Estimate the memory required to store model parameters.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of Transformer blocks
        d_ff: Feed-forward hidden dimension
        dtype_bytes: Bytes per parameter (4 for float32, 2 for float16)
    
    Returns:
        Approximate memory in bytes
    """
    # Parameter count estimation breakdown:
    # Embeddings: vocab_size * d_model
    # Per Layer:
    #   Attention: 4 * d_model^2 (Wq, Wk, Wv, Wo)
    #   FFN: 3 * d_model * d_ff (W1, W2, W3)
    #   RMSNorm: 2 * d_model (ln1, ln2 scale parameters)
    # Final Norm: d_model
    # Output projection: d_model * vocab_size
    
    params = (
        vocab_size * d_model +
        num_layers * (4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model) +
        d_model +
        d_model * vocab_size
    )
    
    return params * dtype_bytes