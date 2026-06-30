"""Deterministic synthesis of hybrid-search columns from a row id.

Used by both the loader (insert_embeddings) and the offline ground-truth
construction script so that every row's user_array / pipeline_id /
pipeline_doc_id / tags are reproducible from the id alone.

Rate markers
============
To make the filtered ground-truth exact and trivially reproducible,
each row's `user_array` includes deterministic marker tokens of the form
``r_<denom>`` when ``row_id % denom == 0``. The filter probe ``user_array
@> ARRAY['r_<denom>']`` then matches exactly ``ceil(N / denom)`` rows.
Likewise ``doc_tags`` carries the same markers keyed by ``doc_id``.
The remaining filler values come from a Zipf-like cardinality so that
GIN indexes have realistic value distributions.
"""

from __future__ import annotations

import hashlib

# Default cardinalities -- tuned to match production traces (array length ~50,
# pipeline cardinality ~100). Rate markers are layered on top.
ARRAY_CARDINALITY = 100_000
ARRAY_FILLER_LEN = 45
PIPELINE_BUCKETS = 1_000
CHUNKS_PER_DOC = 10
DOC_TAGS_CARDINALITY = 10_000
DOC_TAGS_FILLER_LEN = 15

# Each marker hits exactly 1/denom of the corpus.
RATE_DENOMS = (2, 10, 100, 1_000, 10_000)


def _h(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode())
        h.update(b"|")
    return int.from_bytes(h.digest(), "big")


def rate_markers_for_id(row_id: int, denoms: tuple[int, ...] = RATE_DENOMS) -> list[str]:
    """Return marker tokens whose mere presence in user_array fixes the
    filter selectivity at exactly 1/denom."""
    return [f"r_{d}" for d in denoms if row_id % d == 0]


def rate_marker_for_rate(rate: float) -> str:
    """Map a rate in (0, 1] to its corresponding marker token.
    Rate must equal 1/denom for a denom in RATE_DENOMS."""
    for d in RATE_DENOMS:
        if abs(rate - 1.0 / d) < 1e-9:
            return f"r_{d}"
    msg = f"unsupported rate {rate}; supported: {[1/d for d in RATE_DENOMS]}"
    raise ValueError(msg)


def user_array_for(
    row_id: int,
    cardinality: int = ARRAY_CARDINALITY,
    filler_len: int = ARRAY_FILLER_LEN,
) -> list[str]:
    """Return the deterministic TEXT[] for a chunk row: rate markers
    followed by Zipf-like filler tokens."""
    out = list(rate_markers_for_id(row_id))
    seen: set[int] = set()
    i = 0
    while len(out) - len(rate_markers_for_id(row_id)) < filler_len:
        v = _h("ua", row_id, i) % cardinality
        if v not in seen:
            seen.add(v)
            out.append(f"a{v}")
        i += 1
    return out


def pipeline_id_for(row_id: int, buckets: int = PIPELINE_BUCKETS) -> str:
    return f"p{_h('pid', row_id) % buckets}"


def pipeline_doc_id_for(row_id: int, chunks_per_doc: int = CHUNKS_PER_DOC) -> int:
    return row_id // chunks_per_doc


def doc_tags_for(
    doc_id: int,
    cardinality: int = DOC_TAGS_CARDINALITY,
    filler_len: int = DOC_TAGS_FILLER_LEN,
) -> list[str]:
    out = list(rate_markers_for_id(doc_id))
    seen: set[int] = set()
    i = 0
    while len(out) - len(rate_markers_for_id(doc_id)) < filler_len:
        v = _h("tag", doc_id, i) % cardinality
        if v not in seen:
            seen.add(v)
            out.append(f"t{v}")
        i += 1
    return out
