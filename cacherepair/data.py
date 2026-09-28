"""Reconstruct fixed evaluation inputs from the four original dataset sources."""

from __future__ import annotations
import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence
from transformers import AutoTokenizer

QUERY_TEMPLATE = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words.\nQuestion: {question}\nAnswer:"


@dataclass(frozen=True)
class AdaptedRow:
    chunks: tuple[str, ...]
    document_ids: tuple[str, ...]
    question: str
    answers: tuple[str, ...]


def _text(value) -> str:
    return "" if value is None else str(value)


def _decode_container(value):
    """Decode Parquet-converted JSON containers while leaving ordinary text intact."""

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, (list, dict)) else value


def _chunk(title, body) -> str:
    return (f"{_text(title)}\n{_text(body)}").strip()


def _answers(row: Mapping) -> tuple[str, ...]:
    values = []
    raw_answers = row.get("answers")
    if isinstance(raw_answers, list):
        values.extend(raw_answers)
    answer = row.get("answer")
    if isinstance(answer, Mapping):
        for key in ("value", "normalized_value"):
            if answer.get(key) is not None:
                values.append(answer[key])
        for key in ("aliases", "normalized_aliases"):
            aliases = answer.get(key, [])
            if isinstance(aliases, list):
                values.extend(aliases)
    elif answer is not None:
        values.append(answer)
    aliases = row.get("answer_aliases", [])
    if isinstance(aliases, list):
        values.extend(aliases)
    result = []
    seen = set()
    for value in values:
        value = _text(value).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _parallel_context_columns(context: Mapping) -> list[tuple[str, object]]:
    titles = context.get("title", [])
    sentences = context.get("sentences", context.get("text", []))
    if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
        return []
    if not isinstance(sentences, Sequence) or isinstance(sentences, (str, bytes)):
        return []
    return list(zip(titles, sentences))


def _context_pairs(context) -> list[tuple[str, object]]:
    context = _decode_container(context)
    if isinstance(context, Mapping):
        return _parallel_context_columns(context)
    if not isinstance(context, list):
        return []
    pairs = []
    for item in context:
        if isinstance(item, Mapping):
            pairs.append(
                (
                    item.get("title", item.get("id", "")),
                    item.get("sentences", item.get("text", item.get("paragraph_text", ""))),
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((item[0], item[1]))
    return pairs


def _body(value) -> str:
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    return _text(value)


def _from_context_pairs(row: Mapping) -> AdaptedRow:
    pairs = _context_pairs(row.get("context", row.get("contexts", [])))
    if not pairs:
        pairs = _context_pairs(row.get("ctxs", row.get("paragraphs", [])))
    chunks = tuple(_chunk(title, _body(body)) for title, body in pairs)
    doc_ids = tuple(f"context-{index}" for index in range(len(chunks)))
    return AdaptedRow(chunks, doc_ids, _text(row.get("question")), _answers(row))


def adapt_musique(row: Mapping) -> AdaptedRow:
    contexts = row.get("ctxs")
    if contexts is None:
        contexts = row.get("paragraphs", [])
    chunks = []
    ids = []
    for index, item in enumerate(contexts):
        if not isinstance(item, Mapping):
            raise ValueError("MuSiQue contexts must be objects")
        chunks.append(
            _chunk(item.get("title", ""), item.get("text", item.get("paragraph_text", "")))
        )
        ids.append(_text(item.get("id", item.get("idx", f"context-{index}"))))
    return AdaptedRow(tuple(chunks), tuple(ids), _text(row.get("question")), _answers(row))


def adapt_hotpot(row: Mapping) -> AdaptedRow:
    return _from_context_pairs(row)


def _trivia_group(
    group: Mapping,
    title_key: str,
    text_key: str,
    *,
    fallback_title_key: str | None = None,
) -> list[tuple[str, str]]:
    titles = group.get(title_key)
    if titles is None and fallback_title_key is not None:
        titles = group.get(fallback_title_key)
    if titles is None:
        titles = []
    texts = group.get(text_key, [])
    if not isinstance(titles, list) or not isinstance(texts, list):
        raise ValueError("TriviaQA evidence columns must be lists")
    if len(titles) != len(texts):
        raise ValueError("TriviaQA evidence title/context columns must have equal length")
    return [(_text(title), _text(body)) for title, body in zip(titles, texts)]


def adapt_triviaqa(row: Mapping) -> AdaptedRow:
    pairs = []
    entity = row.get("entity_pages", {})
    search = row.get("search_results", {})
    if isinstance(entity, Mapping):
        pairs.extend(
            _trivia_group(
                entity,
                "title",
                "wiki_context",
                fallback_title_key="wiki_title",
            )
        )
    if isinstance(search, Mapping):
        pairs.extend(_trivia_group(search, "title", "search_context"))
    if not pairs:
        pairs = [(title, _body(body)) for title, body in _context_pairs(row.get("context", []))]
    chunks = tuple(_chunk(title, body) for title, body in pairs)
    ids = tuple(f"evidence-{index}" for index in range(len(chunks)))
    return AdaptedRow(chunks, ids, _text(row.get("question")), _answers(row))


def tokenize_adapted_row(
    adapted: AdaptedRow,
    tokenizer,
    *,
    max_chunks: int = 20,
    max_document_tokens: int = 4_096,
    max_query_tokens: int = 128,
) -> dict:
    """Apply one target-independent chunk policy and target-tokenizer caps."""

    selected_chunks = adapted.chunks[:max_chunks]
    selected_ids = adapted.document_ids[:max_chunks]
    budget_per_chunk = max(1, max_document_tokens // len(selected_chunks))
    chunk_token_ids = []
    retained_chunks = []
    retained_ids = []
    used = 0
    for text, document_id in zip(selected_chunks, selected_ids):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=budget_per_chunk,
        )
        ids = ids[: max(0, max_document_tokens - used)]
        if ids:
            chunk_token_ids.append([int(token) for token in ids])
            retained_chunks.append(text)
            retained_ids.append(str(document_id))
            used += len(ids)
        if used >= max_document_tokens:
            break
    if not chunk_token_ids:
        raise ValueError("tokenization removed every document chunk")
    query_text = QUERY_TEMPLATE.format(question=adapted.question)
    query_ids = [
        int(token)
        for token in tokenizer.encode(
            query_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_query_tokens,
        )
    ]
    if not query_ids:
        raise ValueError("tokenization produced an empty query")
    return {
        "ordered_chunks": retained_chunks,
        "document_ids": retained_ids,
        "document_token_ids_by_chunk": chunk_token_ids,
        "query_prompt": query_text,
        "query_token_ids": query_ids,
        "token_counts": {
            "chunks": [len(ids) for ids in chunk_token_ids],
            "document": sum(len(ids) for ids in chunk_token_ids),
            "query": len(query_ids),
        },
    }


CHUNK_TOKENS = 512
RETRIEVED_CHUNKS = 8
MAX_CHUNKS_PER_DOCUMENT = 2
_TERM = re.compile(r"[^\W_]+", flags=re.UNICODE)


def retrieval_terms(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TERM.finditer(str(text)))


def _metadata(document: Mapping) -> str:
    return (
        f"Title: {document.get('title', '')}\n"
        f"Source: {document.get('source', '')}\n"
        f"Published: {document.get('published_at', '')}\n"
        f"Category: {document.get('category', '')}\n"
    )


def chunk_corpus(
    corpus: Sequence[Mapping],
    tokenizer,
    *,
    chunk_tokens: int = CHUNK_TOKENS,
) -> list[dict]:
    """Split every released article into non-overlapping target-token chunks."""

    if not corpus or chunk_tokens <= 0:
        raise ValueError("MultiHop-RAG corpus and chunk size must be nonempty")
    chunks: list[dict] = []
    seen_ids: set[str] = set()
    for document_index, document in enumerate(corpus):
        if not isinstance(document, Mapping):
            raise ValueError("MultiHop-RAG corpus rows must be objects")
        prefix = _metadata(document)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if len(prefix_ids) >= chunk_tokens:
            raise ValueError("MultiHop-RAG document metadata exceeds one chunk")
        body_ids = tokenizer.encode(str(document.get("body", "")), add_special_tokens=False)
        if not body_ids:
            raise ValueError(f"MultiHop-RAG corpus row {document_index} has no body")
        body_budget = chunk_tokens - len(prefix_ids)
        document_id = hashlib.sha256(
            str(document.get("url", document_index)).encode("utf-8")
        ).hexdigest()[:16]
        if document_id in seen_ids:
            raise ValueError("MultiHop-RAG corpus document identity is duplicated")
        seen_ids.add(document_id)
        for chunk_index, start in enumerate(range(0, len(body_ids), body_budget)):
            ids = [*prefix_ids, *body_ids[start : start + body_budget]]
            text = tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not text:
                raise ValueError("MultiHop-RAG token chunk decoded to empty text")
            chunks.append(
                {
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                    "document_id": document_id,
                    "chunk_id": f"{document_id}:{chunk_index}",
                    "title": str(document.get("title", "")),
                    "text": text,
                    "terms": retrieval_terms(text),
                    "token_count": len(ids),
                }
            )
    return chunks


def build_bm25_index(chunks: Sequence[Mapping]) -> dict:
    if not chunks:
        raise ValueError("BM25 index needs at least one chunk")
    frequencies: list[Counter] = []
    document_frequency: Counter = Counter()
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for ordinal, chunk in enumerate(chunks):
        terms = tuple(chunk.get("terms", ()))
        if not terms:
            raise ValueError("BM25 chunk has no retrieval terms")
        counts = Counter(terms)
        frequencies.append(counts)
        document_frequency.update(counts.keys())
        for term, frequency in counts.items():
            postings.setdefault(term, []).append((ordinal, int(frequency)))
        lengths.append(len(terms))
    return {
        "frequencies": frequencies,
        "document_frequency": document_frequency,
        "postings": postings,
        "lengths": lengths,
        "average_length": sum(lengths) / len(lengths),
        "chunk_count": len(chunks),
    }


def retrieve_chunks(
    query: str,
    chunks: Sequence[Mapping],
    index: Mapping,
    *,
    top_k: int = RETRIEVED_CHUNKS,
    max_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """Return a deterministic query-only BM25 ranking with source diversity."""

    query_terms = retrieval_terms(query)
    if not query_terms or top_k <= 0 or max_per_document <= 0:
        raise ValueError("BM25 query and retrieval limits must be nonempty")
    n = int(index["chunk_count"])
    avgdl = float(index["average_length"])
    dfs: Mapping = index["document_frequency"]
    lengths = index["lengths"]
    if len(chunks) != len(index["frequencies"]) or len(chunks) != len(lengths):
        raise ValueError("BM25 index arrays do not align with corpus chunks")
    scores: Counter = Counter()
    query_frequencies = Counter(query_terms)
    for term, query_frequency in query_frequencies.items():
        df = int(dfs.get(term, 0))
        if not df:
            continue
        inverse_document_frequency = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for ordinal, frequency in index["postings"][term]:
            length = lengths[ordinal]
            denominator = frequency + k1 * (1.0 - b + b * length / avgdl)
            scores[ordinal] += (
                query_frequency * inverse_document_frequency * frequency * (k1 + 1.0) / denominator
            )

    def ranking_key(ordinal: int) -> tuple[float, int, int]:
        chunk = chunks[ordinal]
        return (
            -float(scores.get(ordinal, 0.0)),
            int(chunk["document_index"]),
            int(chunk["chunk_index"]),
        )

    ranking = sorted(range(n), key=ranking_key)
    selected: list[dict] = []
    per_document: Counter = Counter()
    for ordinal in ranking:
        chunk = chunks[ordinal]
        document_id = str(chunk["document_id"])
        if per_document[document_id] >= max_per_document:
            continue
        selected.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "document_id": document_id,
                "title": str(chunk["title"]),
                "text": str(chunk["text"]),
                "bm25_score": float(scores.get(ordinal, 0.0)),
            }
        )
        per_document[document_id] += 1
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise ValueError("BM25 retrieval could not satisfy the frozen chunk count")
    return selected


TARGETS = {
    "qwen3": "qwen2.5_3b_instruct",
    "llama8": "llama3.1_8b_instruct",
    "qwen14": "qwen2.5_14b_instruct",
}
DATA = Path(__file__).resolve().parent / "assets/evaluation"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def load_source(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    values = json.loads(path.read_text())
    if not isinstance(values, list):
        raise ValueError(f"Expected a list of source records: {path}")
    return values


def rebalance(record, tokenizer):
    flat = [token for chunk in record["document_token_ids_by_chunk"] for token in chunk]
    count = min(len(flat), 20)
    width, remainder = divmod(len(flat), count)
    windows = []
    start = 0
    for i in range(count):
        stop = start + width + (i < remainder)
        windows.append(flat[start:stop])
        start = stop
    record.update(
        ordered_chunks=[
            tokenizer.decode(window, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for window in windows
        ],
        document_ids=[f"{record['sample_id']}:geometry20-{i:03d}" for i in range(count)],
        document_token_ids_by_chunk=windows,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", choices=["musique", "hotpotqa", "multihoprag", "triviaqa"], required=True
    )
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument(
        "--source",
        nargs="+",
        type=Path,
        required=True,
        help="Original dataset files, in published shard order",
    )
    p.add_argument("--corpus", type=Path, help="Original MultiHop-RAG corpus.json")
    p.add_argument(
        "--qwen-tokenizer",
        required=True,
        help="Qwen2.5-3B-Instruct tokenizer directory or official model ID",
    )
    p.add_argument("--llama-tokenizer", help="Llama-3.1-8B-Instruct tokenizer, required for llama8")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, help="Build the first N fixed requests for a quick check")
    a = p.parse_args()
    if a.target == "llama8" and not a.llama_tokenizer:
        p.error("--llama-tokenizer is required for llama8")
    if a.dataset == "multihoprag" and not a.corpus:
        p.error("--corpus is required for multihoprag")
    if a.limit is not None and not 0 < a.limit <= 500:
        p.error("--limit must be between 1 and 500")
    if a.output.exists():
        p.error("--output already exists; choose a new file")

    ids = [json.loads(line) for line in (DATA / f"{a.dataset}.jsonl").read_text().splitlines()]
    ids = ids[: a.limit] if a.limit else ids
    specification = json.loads((DATA / "construction.json").read_text())[a.dataset]
    sources = [row for path in a.source for row in load_source(path)]
    qtokenizer = AutoTokenizer.from_pretrained(a.qwen_tokenizer, trust_remote_code=False)
    target_tokenizer = (
        AutoTokenizer.from_pretrained(a.llama_tokenizer, trust_remote_code=False)
        if a.target == "llama8"
        else qtokenizer
    )
    if a.dataset == "multihoprag":
        corpus_chunks = chunk_corpus(load_source(a.corpus), qtokenizer)
        retrieval_index = build_bm25_index(corpus_chunks)

    boundary = "CACHEREPAIR_CONTENT_BOUNDARY_PLACEHOLDER"
    prefix, suffix = target_tokenizer.apply_chat_template(
        [{"role": "user", "content": boundary}], tokenize=False, add_generation_prompt=True
    ).split(boundary)
    prefix_ids = target_tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = target_tokenizer.encode(suffix, add_special_tokens=False)
    output = []
    for item in ids:
        row = sources[item["source_row"]]
        question = row["query"] if a.dataset == "multihoprag" else row["question"]
        if hashlib.sha256(question.encode()).hexdigest() != item["question_sha256"]:
            raise ValueError(f"Original source order/content differs at {item['sample_id']}")
        if a.dataset == "multihoprag":
            retrieved = retrieve_chunks(question, corpus_chunks, retrieval_index)
            adapted = AdaptedRow(
                tuple(r["text"] for r in retrieved),
                tuple(r["chunk_id"] for r in retrieved),
                question,
                (row["answer"],),
            )
        else:
            adapted = {
                "musique": adapt_musique,
                "hotpotqa": adapt_hotpot,
                "triviaqa": adapt_triviaqa,
            }[a.dataset](row)
        record = dict(
            sample_id=item["sample_id"],
            dataset=a.dataset,
            source_split=item["sample_id"].split(":")[1],
            source_row=item["source_row"],
            query=question,
            answers=list(adapted.answers),
            source_hashes=copy.deepcopy(specification["source_hashes"]),
            **tokenize_adapted_row(adapted, qtokenizer),
        )
        if a.dataset in {"triviaqa", "multihoprag"}:
            rebalance(record, qtokenizer)
        if a.target == "llama8":
            record.update(
                tokenize_adapted_row(
                    AdaptedRow(
                        tuple(record["ordered_chunks"]),
                        tuple(record["document_ids"]),
                        question,
                        tuple(record["answers"]),
                    ),
                    target_tokenizer,
                )
            )
        record.update(source_id=item["source_id"], selection_rank=item["rank"])
        if a.dataset == "triviaqa":
            record["upstream_question_id"] = row["question_id"]
        record["document_token_ids_by_chunk"][0] = (
            prefix_ids + record["document_token_ids_by_chunk"][0]
        )
        record["query_prompt"] = (
            "\n\nAnswer the question based on the passages. Return only the minimal answer phrase, "
            "with no explanation or extra description. Do not restate the question.\nQuestion: "
            + question
            + "\nAnswer:"
        )
        record["query_token_ids"] = (
            target_tokenizer.encode(record["query_prompt"], add_special_tokens=False) + suffix_ids
        )
        lengths = [len(chunk) for chunk in record["document_token_ids_by_chunk"]]
        record["token_counts"] = dict(
            chunks=lengths, document=sum(lengths), query=len(record["query_token_ids"])
        )
        if digest(record) != item["input_hashes"][a.target]:
            raise ValueError(
                f"Reconstructed input differs from the paper at {item['sample_id']}; check source files and tokenizer revisions"
            )
        output.append(record)
        if len(output) % 50 == 0:
            print(json.dumps({"built": len(output), "total": len(ids)}), flush=True)
    manifest = dict(
        kind="cacherepair_requests_v1",
        dataset=a.dataset,
        target_model=TARGETS[a.target],
        records=output,
        row_count=len(output),
        records_sha256=digest(output),
        generation=dict(temperature=0, max_new_tokens=32),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {"output": str(a.output), "requests": len(output), "all_input_hashes_match": True}
        )
    )


if __name__ == "__main__":
    main()
