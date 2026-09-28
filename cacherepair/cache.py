"""Independent chunk prefill, canonical K coordinates, and global cache assembly."""

from __future__ import annotations
import torch


def rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotate_keys(keys, positions, inv_freq, *, scaling=1.0, inverse=False):
    """Native target RoPE in FP32; the token axis is dimension one."""
    angles = positions.float()[:, None] * inv_freq.float()[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    cos, sin = (cos / scaling, -sin / scaling) if inverse else (cos * scaling, sin * scaling)
    shape = [1, keys.shape[1]] + [1] * (keys.ndim - 3) + [keys.shape[-1]]
    return keys.float() * cos.view(shape) + rotate_half(keys.float()) * sin.view(shape)


def pack_cache(past):
    """HF cache -> [B,N,L,Hkv,2,D], with K then V on the penultimate axis."""
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    keys = torch.stack([pair[0] for pair in past], dim=0)
    values = torch.stack([pair[1] for pair in past], dim=0)
    return torch.stack((keys, values), dim=4).permute(1, 3, 0, 2, 4, 5).contiguous()


def to_hf_cache(global_kv):
    """Global-position packed cache -> Hugging Face DynamicCache."""
    from transformers import DynamicCache

    layers = global_kv.permute(2, 4, 0, 3, 1, 5)
    return DynamicCache.from_legacy_cache(tuple((layer[0], layer[1]) for layer in layers))


def position_inputs(tokens):
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    return dict(
        position_ids=positions[None].expand(tokens.shape[0], -1),
        cache_position=positions,
        attention_mask=torch.ones_like(tokens),
    )


def rerotate_chunk(keys, offset, target):
    """Move locally rotated BF16 keys to their position in the assembled prefix."""
    if offset == 0:
        return keys
    rotary = target.model.rotary_emb
    if target.config.rope_scaling is None:
        angles = float(offset) * rotary.inv_freq.float()
        cos = torch.cat((angles.cos(), angles.cos()))
        sin = torch.cat((angles.sin(), angles.sin()))
    else:
        batch, length = keys.shape[:2]
        positions = torch.arange(length, device=keys.device)[None].expand(batch, -1)
        probe = torch.empty((batch, 1, 1, keys.shape[-1]), device=keys.device, dtype=torch.float32)
        local_cos, local_sin = rotary(probe, positions)
        global_cos, global_sin = rotary(probe, positions + offset)
        scale2 = float(rotary.attention_scaling) ** 2
        cos = (
            global_cos.float() * local_cos.float() + global_sin.float() * local_sin.float()
        ) / scale2
        sin = (
            global_sin.float() * local_cos.float() - global_cos.float() * local_sin.float()
        ) / scale2
        shape = [batch, length] + [1] * (keys.ndim - 3) + [keys.shape[-1]]
        cos, sin = cos.reshape(shape), sin.reshape(shape)
    return (keys.float() * cos + rotate_half(keys.float()) * sin).to(keys.dtype)


def canonicalize(global_kv, target):
    """Undo global target RoPE on K, returning FP32 canonical KV."""
    rotary = target.model.rotary_emb
    positions = torch.arange(global_kv.shape[1], device=global_kv.device)
    keys = rotate_keys(
        global_kv[..., 0, :],
        positions,
        rotary.inv_freq,
        scaling=float(rotary.attention_scaling),
        inverse=True,
    )
    return torch.stack((keys, global_kv[..., 1, :].float()), dim=4)


def materialize(canonical, inv_freq, scaling=1.0, dtype=torch.bfloat16):
    """Apply global target RoPE and cast once into the serving cache dtype."""
    positions = torch.arange(canonical.shape[1], device=canonical.device)
    keys = rotate_keys(canonical[..., 0, :], positions, inv_freq, scaling=scaling)
    return torch.stack((keys, canonical[..., 1, :]), dim=4).to(dtype)


@torch.no_grad()
def compile_stale(target, chunks):
    """Return canonical stale KV, token IDs, and chunk IDs for one request.

    Each chunk is prefilled independently. Local-to-global rerotation retains
    the BF16 rounding used during training before conversion to canonical K.
    """
    device = target.get_input_embeddings().weight.device
    parts, tokens, ids, offset = [], [], [], 0
    for index, chunk in enumerate(chunks):
        chunk = torch.as_tensor(chunk, dtype=torch.long, device=device).reshape(1, -1)
        output = target(chunk, use_cache=True, **position_inputs(chunk))
        kv = pack_cache(output.past_key_values)
        kv[..., 0, :] = rerotate_chunk(kv[..., 0, :], offset, target)
        parts.append(kv)
        tokens.append(chunk)
        ids.append(torch.full_like(chunk, index))
        offset += chunk.shape[1]
    return (
        canonicalize(torch.cat(parts, dim=1), target),
        torch.cat(tokens, dim=1),
        torch.cat(ids, dim=1),
    )


@torch.no_grad()
def make_pair(target, chunks):
    """Build stale/joint supervision on demand for residual-MSE training."""
    stale, tokens, chunk_ids = compile_stale(target, chunks)
    joint = pack_cache(target(tokens, use_cache=True, **position_inputs(tokens)).past_key_values)
    return dict(stale=stale, joint=canonicalize(joint, target), tokens=tokens, chunk_ids=chunk_ids)
