# Training

For each frozen target LLM, independent chunk prefill produces stale KV and a
joint document prefill produces the target KV. Keys are converted to canonical
RoPE coordinates. `model.py` divides the stale input by its per-coordinate RMS,
encodes each layer/head/K-or-V segment, and combines the compressed features
with frozen target token embeddings. Block-causal repair blocks exchange
information within each chunk and from preceding chunks, with stale features
reinjected at every block. The output head predicts the normalized residual.

The loss is the mean squared difference between the normalized prediction and
`(joint - stale) / sigma_delta`. At inference, the predicted residual is
rescaled and added to stale KV, and target-native global RoPE is applied to K.
The released weights include `sigma_stale` and `sigma_delta`.

The supplied training entry point builds cache pairs on demand. Its input is a
JSONL file with one example per line:

```json
{"chunks": [[101, 102, 103], [201, 202, 203]]}
```

Replace these illustrative integers with token IDs from the target tokenizer.
Each example contains its ordered document chunks. The first pass estimates
RMS statistics and saves `statistics.safetensors`; use `--stats PATH` to reuse
those statistics. Each update accumulates gradients over four examples, so
requests can have different token lengths. The optimizer updates only repair
parameters. The frozen target LLM uses BF16; repairer training, canonical KV,
and normalization statistics use FP32, matching the original training path.

The published models use six passes over 50,000 examples: 75,000 optimizer
updates. The learning-rate horizon is 100,000 updates, with 2,000 warmup updates
to `3e-4`, then cosine decay toward `3e-5`. AdamW uses betas `(0.9, 0.95)`, epsilon
`1e-8`, weight decay `0.01`, and gradient norm clipping at `1.0`. Configuration
files retain the architecture and initialization seeds for each target/size.
The initialization seed controls parameter initialization; the training seed
controls the fixed shuffle for each epoch. For numerical reproduction, use the
same PyTorch version, target weights, token inputs, and GPU environment.

The entry point writes training loss and one checkpoint per completed epoch.
Published models can be evaluated directly; training on another corpus produces
a new repairer for that corpus and target LLM.
