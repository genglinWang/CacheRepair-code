"""Generate answers and measure F1/TTFT for the three cache-processing methods."""

from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import statistics
import string
import subprocess
import sys
from time import perf_counter

MODEL_NAMES = {
    "qwen2.5_3b_instruct": "Qwen2.5-3B-Instruct",
    "llama3.1_8b_instruct": "Llama-3.1-8B-Instruct",
    "qwen2.5_14b_instruct": "Qwen2.5-14B-Instruct",
}


def answer_scores(text, answers):
    prediction = text.lstrip("\n").split("\n")[0].strip()
    boolean = re.match(r"^(yes|no)(?:[.,!:;](?:\s|$)|$)", prediction, re.IGNORECASE)
    if boolean:
        prediction = boolean.group(1)

    def normalize(value):
        value = "".join(c for c in value.lower() if c not in string.punctuation)
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())

    prediction = normalize(prediction)
    f1, em = 0.0, 0.0
    for answer in answers or [""]:
        gold = normalize(answer)
        p, g = prediction.split(), gold.split()
        common = sum((Counter(p) & Counter(g)).values())
        if p and g and common:
            precision, recall = common / len(p), common / len(g)
            score = 2 * precision * recall / (precision + recall)
        else:
            score = float(p == g)
        f1, em = max(f1, score), max(em, float(prediction == gold))
    return f1, em


def generate_timed(llm, prompt, sampling):
    """One generation supplies both quality and scheduler-to-first-token time."""
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    sampling.output_kind = RequestOutputKind.CUMULATIVE
    start = perf_counter()
    llm._add_request(TokensPrompt(prompt_token_ids=prompt), sampling)
    first, final = None, None
    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            if first is None and any(len(sequence.token_ids) for sequence in output.outputs):
                first = perf_counter() - start
            if output.finished:
                final = output
    if final is None:
        raise RuntimeError("Generation completed without a final output")
    return final.outputs[0], first if first is not None else perf_counter() - start


def run_all(args):
    args.output.mkdir(parents=True, exist_ok=False)
    for method, size in [
        ("full", "M"),
        ("stale", "M"),
        ("repair", "S"),
        ("repair", "M"),
        ("repair", "L"),
    ]:
        name = f"repair-{size}" if method == "repair" else method
        command = [
            sys.executable,
            "-m",
            "cacherepair.evaluate",
            "--model",
            args.model,
            "--manifest",
            str(args.manifest),
            "--method",
            method,
            "--size",
            size,
            "--weights",
            args.weights,
            "--output",
            str(args.output / name),
            "--gpu",
            str(args.gpu),
            "--seed",
            str(args.seed),
        ]
        if args.limit:
            command += ["--limit", str(args.limit)]
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Matching target LLM: local directory or official HF ID"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Output of python -m cacherepair.data"
    )
    parser.add_argument(
        "--method", choices=["full", "stale", "repair", "all"], default="repair"
    )
    parser.add_argument(
        "--weights",
        default="gwang3456/cacherepair",
        help="HF repair repository or downloaded model root",
    )
    parser.add_argument("--size", choices=["S", "M", "L"], default="M")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="vLLM generation seed")
    parser.add_argument("--limit", type=int, help="Use the first N requests")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.method == "all":
        run_all(args)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output / "runtime"
    runtime.mkdir()
    root = str(Path(__file__).resolve().parents[1])
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        VLLM_USE_V1="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_NO_USAGE_STATS="1",
        CACHEREPAIR_VLLM="1",
        TOKENIZERS_PARALLELISM="false",
        TRITON_CACHE_DIR=str(runtime / "triton"),
        PYTHONPATH=os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]),
    )
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"][: args.limit] if args.limit else manifest["records"]
    if not records:
        parser.error("The request manifest is empty")
    manifest_path = runtime / "requests.json"
    manifest_path.write_text(json.dumps({**manifest, "records": records}))

    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from .cache import compile_stale
    from .model import checkpoint_directory, load_repairer

    compiler, checkpoint = None, None
    engine_args = dict(
        model=args.model,
        dtype="bfloat16",
        max_model_len=4352,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        num_gpu_blocks_override=512,
        max_num_batched_tokens=4352,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=args.seed,
    )
    if args.method != "full":
        from .connector import register, bind_embedding, preload, evict, last_stats, next_bucket

        register()
        compiler = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            .to("cuda")
            .eval()
        )
        compiler.requires_grad_(False)
        if args.method == "repair":
            checkpoint = checkpoint_directory(
                args.weights, MODEL_NAMES[manifest["target_model"]] + "/" + args.size
            )
            # Check the target geometry and native rotary frequencies once when loading.
            temporary = load_repairer(checkpoint, compiler)
            del temporary
        rotary = compiler.model.rotary_emb
        options = dict(
            manifest=str(manifest_path),
            checkpoint=str(checkpoint) if checkpoint else None,
            inv_freq=rotary.inv_freq.float().cpu().tolist(),
            rope_scaling=float(rotary.attention_scaling),
        )
        engine_args["kv_transfer_config"] = KVTransferConfig(
            kv_connector="CacheRepairConnector",
            kv_role="kv_both",
            kv_connector_extra_config=options,
        )
    llm = LLM(**engine_args)

    def rpc(function, **kwargs):
        return llm.collective_rpc(function, kwargs=kwargs)[0]

    if args.method == "repair":
        rpc(bind_embedding)

    def execute(record, max_tokens):
        document = [token for chunk in record["document_token_ids_by_chunk"] for token in chunk]
        prompt = document + record["query_token_ids"]
        if compiler is not None:
            stale, tokens, chunk_ids = compile_stale(
                compiler, record["document_token_ids_by_chunk"]
            )
            payload = runtime / "host-cache.pt"
            torch.save(
                dict(
                    keys=stale[..., 0, :].to(torch.bfloat16).cpu(),
                    values=stale[..., 1, :].to(torch.bfloat16).cpu(),
                    tokens=tokens.cpu(),
                    chunk_ids=chunk_ids.cpu(),
                ),
                payload,
            )
            del stale, tokens, chunk_ids
            rpc(preload, payload_path=str(payload))
            payload.unlink()
            torch.cuda.synchronize()
        answer, elapsed = generate_timed(
            llm, prompt, SamplingParams(temperature=0, max_tokens=max_tokens, seed=args.seed)
        )
        telemetry = rpc(last_stats) if compiler is not None else {}
        if compiler is not None:
            expected = len(document) // 16 * 16
            if telemetry.get("external_tokens") != expected:
                raise RuntimeError(
                    "External KV cache was not loaded for the full document prefix"
                )
            rpc(evict)
        f1, em = answer_scores(answer.text, record["answers"])
        return dict(
            sample_id=record["sample_id"],
            text=answer.text,
            token_ids=list(answer.token_ids),
            f1=f1,
            em=em,
            ttft_s=elapsed,
            telemetry=telemetry,
            document_tokens=len(document),
            query_tokens=len(record["query_token_ids"]),
        )

    warmups = {}
    for record in records:
        bucket = next_bucket(record["token_counts"]["document"]) if compiler is not None else 0
        warmups.setdefault(bucket, record)
    for bucket, record in sorted(warmups.items()):
        print(json.dumps({"warmup_bucket": bucket}), flush=True)
        execute(record, 1)
    config = dict(
        method=args.method,
        size=args.size if checkpoint else None,
        target_model=manifest["target_model"],
        dataset=manifest["dataset"],
        target_model_path=args.model,
        checkpoint=str(checkpoint) if checkpoint else None,
        requests=len(records),
        max_new_tokens=32,
        seed=args.seed,
        concurrency=1,
        input_records_sha256=manifest.get("records_sha256"),
        timing="Scheduler submission through first output token, including H2D, repair and global RoPE",
    )
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    with (args.output / "rows.jsonl").open("x", buffering=1) as stream:
        for record in records:
            row = execute(record, 32)
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            print(json.dumps({"completed": len(rows), "total": len(records)}), flush=True)
    summary = dict(
        status="complete",
        requests=len(rows),
        f1=statistics.mean(r["f1"] for r in rows),
        em=statistics.mean(r["em"] for r in rows),
        p50_ttft_ms=1000 * statistics.median(r["ttft_s"] for r in rows),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
