# Data preparation

The repository provides fixed request identities and construction parameters
for 500 requests from each downstream dataset. Obtain question and document
content directly from the original sources under their access terms.

| Dataset | Source | Required files |
|---|---|---|
| MuSiQue | [Official repository](https://github.com/stonybrooknlp/musique) | `musique_ans_v1.0_dev.jsonl` from the answerable v1.0 release |
| HotpotQA | [Official dataset mirror](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | `distractor/validation-00000-of-00001.parquet` |
| MultiHop-RAG | [Official repository](https://github.com/yixuantt/MultiHop-RAG) | `dataset/MultiHopRAG.json` and `dataset/corpus.json` |
| TriviaQA | [Official dataset mirror](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | All four `rc/validation-*.parquet` shards in filename order |

For the Parquet datasets, the Hugging Face client can download just the required
validation files:

```python
from huggingface_hub import snapshot_download
snapshot_download("hotpotqa/hotpot_qa", repo_type="dataset",
                  allow_patterns="distractor/validation-*.parquet", local_dir="sources/hotpotqa")
snapshot_download("mandarjoshi/trivia_qa", repo_type="dataset",
                  allow_patterns="rc/validation-*.parquet", local_dir="sources/triviaqa")
```

Build one manifest per target and dataset. The Qwen2.5-3B tokenizer defines the
initial document layout for all targets; Llama inputs are then tokenized with
Llama-3.1-8B. Both Qwen sizes use the same input token IDs. Examples:

```bash
python -m cacherepair.data --dataset hotpotqa --target qwen14 \
  --source sources/hotpotqa/distractor/validation-00000-of-00001.parquet \
  --qwen-tokenizer Qwen/Qwen2.5-3B-Instruct --output inputs/qwen14/hotpotqa.json

python -m cacherepair.data --dataset multihoprag --target qwen3 \
  --source sources/MultiHopRAG.json --corpus sources/corpus.json \
  --qwen-tokenizer Qwen/Qwen2.5-3B-Instruct --output inputs/qwen3/multihoprag.json

python -m cacherepair.data --dataset triviaqa --target llama8 \
  --source sources/triviaqa/rc/validation-*.parquet \
  --qwen-tokenizer Qwen/Qwen2.5-3B-Instruct \
  --llama-tokenizer meta-llama/Llama-3.1-8B-Instruct --output inputs/llama8/triviaqa.json
```

`cacherepair/assets/evaluation/` stores the ID lists and source-file hashes.
The builder checks each reconstructed input against its saved identity. If a
source or tokenizer revision changes the input, it identifies the affected
request. The output manifest contains the locally reconstructed text and tokens
needed for evaluation.

MuSiQue and HotpotQA retain the supplied context order. TriviaQA uses entity-page
evidence followed by search evidence. MultiHop-RAG retrieves eight chunks with
query-only BM25, with at most two chunks per source article. The input budget is
4,096 document tokens before chat formatting. MultiHop-RAG and TriviaQA are
split into 20 contiguous token windows; the chat prefix is added to the first
chunk and the answer instruction follows the document prefix.

The fixed IDs follow the paper's content screening against the repair-training
corpus: evidence-sentence matches, shared 50-token context spans, and resolved
question provenance. All methods use the same selected requests. Training
sources are ELI5, FEVER, Natural Questions, Wizard of Wikipedia, T-REx, and
Structured Zeroshot through [CoRAG/KILT](https://huggingface.co/datasets/corag/kilt)
and [its passage corpus](https://huggingface.co/datasets/corag/kilt-corpus).
