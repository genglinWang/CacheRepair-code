"""Train a repairer with on-demand stale/joint KV pairs and normalized MSE."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import torch
from transformers import AutoModelForCausalLM
from .cache import make_pair
from .model import CacheRepair, RepairConfig, TargetConfig, save_repairer


def learning_rate(step, horizon):
    warmup = max(1, round(horizon * 0.02))
    if step <= warmup:
        return 3e-4 * step / warmup
    progress = (step - warmup) / (horizon - warmup)
    return 3e-5 + (3e-4 - 3e-5) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def compute_stats(target, rows):
    """Token-weighted RMS of canonical stale KV and its residual to joint KV."""
    sums, count = {}, 0
    for index, row in enumerate(rows):
        pair = make_pair(target, row["chunks"])
        for name, value in [
            ("sigma_stale", pair["stale"]),
            ("sigma_delta", pair["joint"] - pair["stale"]),
        ]:
            term = value.double().square().sum(dim=(0, 1)).cpu()
            sums[name] = sums.get(name, 0) + term
        count += pair["tokens"].numel()
        if (index + 1) % 100 == 0:
            print(
                json.dumps({"statistics_examples": index + 1, "total": len(rows)}), flush=True
            )
    return {
        name: (value / count).clamp_min(1e-12).sqrt().float() for name, value in sums.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Frozen target LLM, as a local path or HF ID"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="JSONL: one {chunks: [[token IDs], ...]} per example",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="One model configuration from configs/"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, help="Previously computed statistics.safetensors")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--schedule-epochs", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.epochs <= args.schedule_epochs:
        parser.error("epochs must be positive and no greater than schedule-epochs")
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Training data must contain at least one example")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads(args.config.read_text())
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    target.requires_grad_(False)
    geometry = TargetConfig.from_hf(target.config)
    if geometry != TargetConfig(**settings["target_config"]):
        raise ValueError("Training configuration and target LLM geometry differ")
    from safetensors.torch import load_file, save_file

    stats = load_file(str(args.stats)) if args.stats else compute_stats(target, rows)
    save_file(stats, str(args.output / "statistics.safetensors"))
    random.seed(settings["training"]["initialization_seed"])
    torch.manual_seed(settings["training"]["initialization_seed"])
    model = (
        CacheRepair(
            geometry,
            RepairConfig.from_dict(settings["repair_config"]),
            target.get_input_embeddings(),
        )
        .to("cuda")
        .train()
    )
    model.sigma_stale.copy_(stats["sigma_stale"])
    model.sigma_delta.copy_(stats["sigma_delta"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    batch_size = 4
    horizon = math.ceil(len(rows) / batch_size) * args.schedule_epochs
    step = 0
    seed = settings["training"]["seed"]
    with (args.output / "loss.jsonl").open("x", buffering=1) as log:
        for epoch in range(args.epochs):
            order = list(range(len(rows)))
            random.Random(seed * 1_000_003 + epoch * 10_007).shuffle(order)
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                step += 1
                lr = learning_rate(step, horizon)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for index in batch:
                    pair = make_pair(target, rows[index]["chunks"])
                    loss = model.loss(**pair)
                    (loss / len(batch)).backward()
                    loss_sum += float(loss.detach())
                    del pair, loss
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                entry = dict(
                    step=step, epoch=epoch + 1, mse=loss_sum / len(batch), learning_rate=lr
                )
                log.write(json.dumps(entry) + "\n")
                if step % 25 == 0:
                    print(json.dumps(entry), flush=True)
            save_repairer(
                args.output / f"checkpoint-{step}",
                model,
                target_model_id=settings["target_model_id"],
                training=dict(
                    checkpoint_epoch=epoch + 1,
                    checkpoint_step=step,
                    examples_per_epoch=len(rows),
                    seed=seed,
                    effective_document_batch=batch_size,
                    learning_rate_horizon_updates=horizon,
                ),
            )


if __name__ == "__main__":
    main()
