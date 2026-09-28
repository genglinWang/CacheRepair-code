"""CacheRepair: stale features, block-causal repair blocks, and residual KV head."""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


@dataclass
class TargetConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    n_attention_heads: int
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 32768

    @property
    def n_segments(self):
        return self.n_layers * self.n_kv_heads * 2

    @classmethod
    def from_hf(cls, config):
        return cls(
            config.num_hidden_layers,
            config.num_key_value_heads,
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            config.hidden_size,
            config.num_attention_heads,
            config.rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )


@dataclass
class RepairConfig:
    d_b: int = 512
    n_blocks: int = 6
    n_heads: int = 8
    d_seg: int = 16
    mlp_ratio: float = 3.0
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, values):
        return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})


class StaleEncoder(nn.Module):
    def __init__(self, target, segment_width, width):
        super().__init__()
        self.W = nn.Parameter(torch.randn(target.n_segments, target.head_dim, segment_width) * 0.02)
        self.b = nn.Parameter(torch.zeros(target.n_segments, segment_width))
        self.out = nn.Linear(target.n_segments * segment_width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, cache):
        segments = torch.einsum("bnsd,sdr->bnsr", cache, self.W) + self.b
        return self.norm(self.out(segments.flatten(2)))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_b // config.n_heads
        self.qkv = nn.Linear(config.d_b, 3 * config.d_b, bias=False)
        self.proj = nn.Linear(config.d_b, config.d_b, bias=False)
        self.use_flex = False
        self._compiled = {}

    def forward(self, hidden, mask, cos, sin, block_mask):
        batch, length, width = hidden.shape
        q, k, v = [
            x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for x in self.qkv(hidden).chunk(3, dim=-1)
        ]

        def rotate(x):
            first, second = x.chunk(2, dim=-1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

        q, k = rotate(q), rotate(k)
        if self.use_flex:
            from torch.nn.attention.flex_attention import flex_attention

            key = (q.device, q.dtype, q.shape)
            if key not in self._compiled:

                def attend(q, k, v, block_mask):
                    return flex_attention(
                        q, k, v, block_mask=block_mask, kernel_options={"PRESCALE_QK": False}
                    )

                self._compiled[key] = torch.compile(attend, dynamic=True)
            out = self._compiled[key](q, k, v, block_mask)
        else:
            additive = torch.zeros(mask.shape, device=q.device, dtype=q.dtype)
            additive.masked_fill_(~mask, torch.finfo(q.dtype).min)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive)
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class RepairBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.d_b
        self.ln1 = nn.LayerNorm(width, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * config.mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(width * config.mlp_ratio), width),
        )
        self.ada = nn.Parameter(torch.zeros(6 * width))

    def forward(self, hidden, mask, cos, sin, block_mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada.chunk(6, dim=-1)
        norm = self.ln1(hidden) * (1 + scale1) + shift1
        hidden = hidden + gate1 * self.attn(norm, mask, cos, sin, block_mask)
        return hidden + gate2 * self.mlp(self.ln2(hidden) * (1 + scale2) + shift2)


class CacheRepair(nn.Module):
    """Residual reconstruction for canonical KV shaped [B,N,L,Hkv,2,D]."""

    def __init__(self, target_config, repair_config, embedding):
        super().__init__()
        self.target_config, self.config = target_config, repair_config
        c, t = repair_config, target_config
        self.stale_enc = StaleEncoder(t, c.d_seg, c.d_b)
        self.tok_embed = embedding
        self.tok_embed.requires_grad_(False)
        self.tok_proj = nn.Linear(embedding.embedding_dim, c.d_b)
        self.fuse = nn.Linear(2 * c.d_b, c.d_b)
        self.blocks = nn.ModuleList([RepairBlock(c) for _ in range(c.n_blocks)])
        with torch.random.fork_rng(devices=[]):
            self.stale_reinject = nn.ModuleList([nn.Linear(c.d_b, c.d_b) for _ in self.blocks])
        for projection in self.stale_reinject:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.ln_f = nn.LayerNorm(c.d_b)
        self.v_head = nn.Linear(c.d_b, t.n_segments * t.head_dim)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)
        shape = (t.n_layers, t.n_kv_heads, 2, t.head_dim)
        self.register_buffer("sigma_delta", torch.ones(shape))
        self.register_buffer("sigma_stale", torch.ones(shape))
        self._rope_cache = {}

    def use_serving_attention(self):
        """Paper policy: FlexAttention followed by three exact-mask blocks."""
        for index, block in enumerate(self.blocks):
            block.attn.use_flex = index < len(self.blocks) - 3

    def forward(self, stale, tokens, chunk_ids, *, bucket=None, block_mask=None):
        """Predict the residual normalized by its training RMS."""
        batch, valid = tokens.shape
        normalized = stale / self.sigma_stale
        if bucket is not None and bucket > valid:
            zeros = normalized.new_zeros(batch, bucket - valid, *normalized.shape[2:])
            normalized = torch.cat((normalized, zeros), dim=1)
            tokens = F.pad(tokens, (0, bucket - valid))
            chunk_ids = F.pad(chunk_ids, (0, bucket - valid), value=-1)
        length = tokens.shape[1]
        mask = (chunk_ids.unsqueeze(-1) >= chunk_ids.unsqueeze(-2)).unsqueeze(1)
        if length > valid:
            positions = torch.arange(length, device=tokens.device)
            real = positions < valid
            mask = mask & real.view(1, 1, length, 1) & real.view(1, 1, 1, length)
            mask |= (
                torch.eye(length, device=tokens.device, dtype=torch.bool) & (~real).view(length, 1)
            ).view(1, 1, length, length)
        with torch.no_grad():
            token_features = self.tok_embed(tokens).to(self.tok_proj.weight.dtype)
        t, c = self.target_config, self.config
        stale_features = self.stale_enc(normalized.reshape(batch, length, t.n_segments, t.head_dim))
        hidden = self.fuse(torch.cat((stale_features, self.tok_proj(token_features)), dim=-1))
        rope_key = (hidden.device, hidden.dtype)
        use_flex = any(block.attn.use_flex for block in self.blocks)
        if not use_flex or rope_key not in self._rope_cache:
            head_dim = c.d_b // c.n_heads
            frequency = 1.0 / (
                c.rope_theta
                ** (torch.arange(0, head_dim, 2, device=hidden.device).float() / head_dim)
            )
            positions = torch.arange(32768 if use_flex else length, device=hidden.device).float()
            angles = positions[:, None] * frequency[None, :]
            tables = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
            if use_flex:
                self._rope_cache[rope_key] = tables
        else:
            tables = self._rope_cache[rope_key]
        cos, sin = [table[:length].view(1, 1, length, -1) for table in tables]
        if use_flex and block_mask is None:
            block_mask = self.prepare_block_mask(chunk_ids, valid_len=valid)
        for projection, block in zip(self.stale_reinject, self.blocks):
            hidden = block(hidden + projection(stale_features), mask, cos, sin, block_mask)
        return self.v_head(self.ln_f(hidden))[:, :valid].view(
            batch, valid, t.n_layers, t.n_kv_heads, 2, t.head_dim
        )

    @torch.no_grad()
    def repair(self, stale, tokens, chunk_ids, **kwargs):
        return stale + self(stale, tokens, chunk_ids, **kwargs) * self.sigma_delta

    def loss(self, stale, joint, tokens, chunk_ids):
        prediction = self(stale, tokens, chunk_ids)
        target = (joint - stale) / self.sigma_delta
        return (prediction.float() - target.float()).square().mean()

    def prepare_block_mask(self, chunk_ids: torch.Tensor, valid_len: int | None = None):
        """Construct sparse block indices for the block-causal attention mask."""
        if chunk_ids.shape[0] != 1:
            raise ValueError("flex_block_causal currently supports batch=1")
        n = int(chunk_ids.shape[1])
        valid_len = n if valid_len is None else int(valid_len)
        if not 0 < valid_len <= n:
            raise ValueError(f"valid Flex length {valid_len} is outside [1, {n}]")
        try:
            from torch.nn.attention.flex_attention import BlockMask
        except ImportError as exc:
            raise ImportError(
                "flex_block_causal requires PyTorch 2.5+ with FlexAttention."
            ) from exc
        block_size = 128
        if n % block_size:
            raise ValueError(
                f"analytic Flex BlockMask requires a {block_size}-aligned bucket, got {n}"
            )
        cids = chunk_ids[0].to(dtype=torch.int32)
        n_blocks = n // block_size
        cids_by_block = cids.view(n_blocks, block_size)
        positions = torch.arange(n, device=cids.device).view(n_blocks, block_size)
        real = positions < valid_len
        has_real = real.any(dim=-1)
        all_real = real.all(dim=-1)
        minimum = torch.where(real, cids_by_block, torch.iinfo(torch.int32).max).amin(dim=-1)
        maximum = torch.where(real, cids_by_block, -1).amax(dim=-1)
        full_tiles = all_real[:, None] & all_real[None, :] & (minimum[:, None] >= maximum[None, :])
        any_real_pair = (
            has_real[:, None] & has_real[None, :] & (maximum[:, None] >= minimum[None, :])
        )
        padded_self = (~all_real)[:, None] & torch.eye(
            n_blocks, dtype=torch.bool, device=cids.device
        )
        partial_tiles = (any_real_pair | padded_self) & ~full_tiles

        def ordered_indices(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            columns = torch.arange(n_blocks, dtype=torch.int32, device=tiles.device).expand(
                n_blocks, n_blocks
            )
            keys = columns + (~tiles).to(dtype=torch.int32) * n_blocks
            indices = torch.argsort(keys, dim=-1).to(dtype=torch.int32)
            counts = tiles.sum(dim=-1, dtype=torch.int32)
            return (counts[None, None], indices[None, None])

        kv_num_blocks, kv_indices = ordered_indices(partial_tiles)
        full_kv_num_blocks, full_kv_indices = ordered_indices(full_tiles)

        def mask_mod(b, h, q_idx, kv_idx):
            q_real = q_idx < valid_len
            kv_real = kv_idx < valid_len
            real_allow = q_real & kv_real & (cids[q_idx] >= cids[kv_idx])
            pad_self = ~q_real & (q_idx == kv_idx)
            return real_allow | pad_self

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=block_size,
            mask_mod=mask_mod,
            seq_lengths=(n, n),
        )


def checkpoint_directory(path_or_repo, subfolder=None):
    path = Path(path_or_repo)
    if path.is_dir():
        return path / subfolder if subfolder else path
    from huggingface_hub import hf_hub_download

    files = [
        hf_hub_download(path_or_repo, name, subfolder=subfolder)
        for name in ("config.json", "model.safetensors")
    ]
    return Path(files[0]).parent


def load_repairer(path_or_repo, target_llm, *, subfolder=None):
    """Load published weights and share the matching frozen target embedding."""
    directory = checkpoint_directory(path_or_repo, subfolder)
    settings = json.loads((directory / "config.json").read_text())
    geometry = TargetConfig.from_hf(target_llm.config)
    if geometry != TargetConfig(**settings["target_config"]):
        raise ValueError("The checkpoint requires its matching target LLM")
    frequency = target_llm.model.rotary_emb.inv_freq.detach().float().cpu().contiguous()
    expected = settings.get("target_rope_spec", {}).get("inv_freq_fp32_sha256")
    if expected and hashlib.sha256(frequency.numpy().tobytes()).hexdigest() != expected:
        raise ValueError("Target rotary frequencies differ from the checkpoint")
    return load_with_embedding(directory, target_llm.get_input_embeddings())


def load_with_embedding(directory, embedding, *, dtype=None):
    settings = json.loads((Path(directory) / "config.json").read_text())
    model = CacheRepair(
        TargetConfig(**settings["target_config"]),
        RepairConfig.from_dict(settings["repair_config"]),
        embedding,
    )
    model.to(device=embedding.weight.device, dtype=dtype)
    result = model.load_state_dict(
        load_file(str(Path(directory) / "model.safetensors")), strict=False
    )
    if result.missing_keys != ["tok_embed.weight"] or result.unexpected_keys:
        raise ValueError(f"Checkpoint architecture differs: {result}")
    return model.eval()


def save_repairer(directory, model, *, target_model_id=None, training=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().cpu().contiguous()
        for k, v in model.state_dict().items()
        if k != "tok_embed.weight"
    }
    save_file(state, str(directory / "model.safetensors"))
    settings = dict(
        format="cacherepair-safetensors-v1",
        target_model_id=target_model_id,
        target_config=asdict(model.target_config),
        repair_config=asdict(model.config),
        key_coordinates="canonical",
        training=training or {},
    )
    (directory / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
