"""vLLM 0.8.5 connector: pinned-host KV -> repair -> paged target cache."""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from .cache import materialize
from .model import load_with_embedding

_RUNNER = None


def next_bucket(length):
    return max(128, 1 << (length - 1).bit_length())


def register():
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    if "CacheRepairConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "CacheRepairConnector", "cacherepair.connector", "CacheRepairConnector"
        )


def bind_embedding(worker):
    weight = worker.model_runner.model.model.embed_tokens.weight
    embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)
    _RUNNER.model = load_with_embedding(_RUNNER.checkpoint, embedding, dtype=torch.bfloat16)
    _RUNNER.model.use_serving_attention()
    return True


def preload(worker, payload_path):
    # The payload is produced locally by evaluate.py, before the timed interval.
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    _RUNNER.host = {name: value.contiguous().pin_memory() for name, value in payload.items()}


def evict(worker):
    _RUNNER.host = None


def last_stats(worker):
    return _RUNNER.stats


class PrefixRunner:
    def __init__(self, options):
        self.checkpoint = options.get("checkpoint")
        self.model = None
        self.host = None
        self.stats = {}
        self.copy_stream = torch.cuda.Stream()
        self.inv_freq = torch.tensor(options["inv_freq"], device="cuda", dtype=torch.float32)
        self.scaling = options["rope_scaling"]

    @torch.no_grad()
    def repair(self):
        host = self.host
        if host is None:
            raise RuntimeError("The request's host cache must be preloaded before generation")
        events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
        (
            copy_start,
            copy_done,
            mask_start,
            mask_done,
            repair_start,
            repair_done,
            rope_start,
            rope_done,
        ) = events
        with torch.cuda.stream(self.copy_stream):
            copy_start.record()
            keys = host["keys"].to("cuda", non_blocking=True)
            values = host["values"].to("cuda", non_blocking=True)
            copy_done.record()
        tokens = host["tokens"].to("cuda", non_blocking=True)
        chunk_ids = host["chunk_ids"].to("cuda", non_blocking=True)
        needs_repair = self.model is not None and int(host["chunk_ids"].max()) > 0
        bucket = next_bucket(tokens.shape[1])
        mask_start.record()
        mask = None
        if needs_repair:
            padded = F.pad(chunk_ids, (0, bucket - tokens.shape[1]), value=-1)
            mask = self.model.prepare_block_mask(padded, valid_len=tokens.shape[1])
        mask_done.record()
        torch.cuda.current_stream().wait_event(copy_done)
        stale = torch.stack((keys, values), dim=4)
        repair_start.record()
        fixed = (
            self.model.repair(stale, tokens, chunk_ids, bucket=bucket, block_mask=mask)
            if needs_repair
            else stale
        )
        repair_done.record()
        rope_start.record()
        fixed = materialize(fixed, self.inv_freq, self.scaling)
        rope_done.record()
        torch.cuda.synchronize()
        self.stats = dict(
            h2d_ms=copy_start.elapsed_time(copy_done),
            mask_ms=mask_start.elapsed_time(mask_done),
            repair_ms=repair_start.elapsed_time(repair_done),
            rope_ms=rope_start.elapsed_time(rope_done),
        )
        return fixed


@dataclass
class RequestCache:
    slots: torch.Tensor
    external_tokens: int


class CacheMetadata(KVConnectorMetadata):
    def __init__(self, requests):
        self.requests = requests


class CacheRepairConnector(KVConnectorBase_V1):
    """Single-request, block-aligned cache reuse with the vLLM V1 scheduler."""

    def __init__(self, vllm_config, role):
        super().__init__(vllm_config, role)
        options = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.block_size = vllm_config.cache_config.block_size
        records = json.loads(Path(options["manifest"]).read_text())["records"]
        self.prefix_lengths = {}
        for record in records:
            document = [token for chunk in record["document_token_ids_by_chunk"] for token in chunk]
            prompt = tuple(document + record["query_token_ids"])
            self.prefix_lengths[prompt] = len(document)
        self.pending = {}
        self.runner = None
        if role == KVConnectorRole.WORKER:
            global _RUNNER
            self.runner = _RUNNER = PrefixRunner(options)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        length = self.prefix_lengths.get(tuple(request.prompt_token_ids), 0)
        external = length // self.block_size * self.block_size
        return max(0, external - num_computed_tokens)

    def update_state_after_alloc(self, request, num_external_tokens):
        if num_external_tokens > 0:
            self.pending[request.request_id] = num_external_tokens

    def build_connector_meta(self, scheduler_output):
        requests = []
        for request in scheduler_output.scheduled_new_reqs:
            external = self.pending.pop(request.req_id, 0)
            if external:
                blocks = torch.tensor(request.block_ids, dtype=torch.long)
                slots = (
                    blocks[:, None] * self.block_size + torch.arange(self.block_size)[None]
                ).flatten()[:external]
                requests.append(RequestCache(slots, external))
        return CacheMetadata(requests)

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CacheMetadata) or self.runner is None:
            return
        for request in metadata.requests:
            fixed = self.runner.repair()
            start, done = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            slots = request.slots.to(fixed.device, non_blocking=True)
            n = request.external_tokens
            for index, layer in enumerate(forward_context.no_compile_layers.values()):
                cache = layer.kv_cache[forward_context.virtual_engine]
                destination = cache.reshape(2, cache.shape[1] * cache.shape[2], -1)
                source = fixed[0, :n, index].permute(2, 0, 1, 3).reshape(2, n, -1)
                destination[:, slots] = source
            done.record()
            torch.cuda.synchronize()
            self.runner.stats.update(write_ms=start.elapsed_time(done), external_tokens=n)

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass
