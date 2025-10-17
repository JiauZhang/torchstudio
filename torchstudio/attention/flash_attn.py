from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from typing import Optional
from torch import Tensor

def flash_attn_func(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0,
    is_causal: bool = False, softmax_scale: Optional[float] = None, enable_gqa: bool = False,
) -> Tensor:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
            scale=softmax_scale, enable_gqa=enable_gqa,
        )
