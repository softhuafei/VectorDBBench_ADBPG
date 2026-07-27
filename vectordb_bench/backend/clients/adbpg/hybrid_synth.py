"""Deterministic synthesis of hybrid-search columns from a row id.

Used by both the loader (insert_embeddings) and the offline ground-truth
construction script so that every row's unified filter columns and the
JOIN-side pipeline_doc_id / tags are reproducible from the id alone.

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
CHUNKS_PER_DOC = 10
DOC_TAGS_CARDINALITY = 10_000
DOC_TAGS_FILLER_LEN = 15

# Each marker hits exactly 1/denom of the corpus.
RATE_DENOMS = (2, 10, 100, 1_000, 10_000)

# RBO boundary experiments focus on medium/high selectivities. Markers are
# nested: a row at percentile 1,500 belongs to every threshold above 1,500.
UNIFIED_RATE_THRESHOLDS = (
    10,
    100,
    500,
    1_000,
    2_000,
    2_200,
    2_500,
    3_000,
    4_000,
    5_000,
    5_500,
    6_000,
    7_000,
    8_000,
    9_000,
)


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


def filter_percentile_for(row_id: int) -> int:
    """Stable percentile bucket in [0, 10000), independent of Python hash."""
    return _h("filter-v1", row_id) % 10_000


def unified_threshold_for_rate(rate: float) -> int:
    threshold = int(round(rate * 10_000))
    if not 0 < threshold <= 10_000 or abs(rate - threshold / 10_000) > 1e-9:
        raise ValueError(f"rate must be representable in basis points and in (0, 1], got {rate}")
    return threshold


def unified_rate_marker_for_rate(rate: float) -> str:
    threshold = unified_threshold_for_rate(rate)
    if threshold not in UNIFIED_RATE_THRESHOLDS:
        supported = [threshold / 10_000 for threshold in UNIFIED_RATE_THRESHOLDS]
        raise ValueError(f"unsupported unified rate {rate}; supported: {supported}")
    return f"r_{threshold}bp"


def unified_rate_markers_for_id(row_id: int) -> list[str]:
    percentile = filter_percentile_for(row_id)
    return [f"r_{threshold}bp" for threshold in UNIFIED_RATE_THRESHOLDS if percentile < threshold]


def unified_user_array_for(
    row_id: int,
    cardinality: int = ARRAY_CARDINALITY,
    filler_len: int = ARRAY_FILLER_LEN,
) -> list[str]:
    markers = unified_rate_markers_for_id(row_id)
    out = list(markers)
    seen: set[int] = set()
    # One cryptographic seed per row is enough for deterministic benchmark
    # filler. SplitMix64 then produces a fast, uniform sequence, avoiding 45
    # Blake2 calls per row on 10M-scale loads.
    state = _h("unified-ua-v2", row_id)
    mask64 = (1 << 64) - 1
    while len(out) - len(markers) < filler_len:
        state = (state + 0x9E3779B97F4A7C15) & mask64
        mixed = state
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask64
        value = (mixed ^ (mixed >> 31)) % cardinality
        if value not in seen:
            seen.add(value)
            out.append(f"a{value}")
    return out


def unified_payload_for(row_id: int) -> dict[str, list[str]]:
    return {"rates": unified_rate_markers_for_id(row_id)}


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
