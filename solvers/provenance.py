"""Canonical provenance encodings shared by production and benchmark paths."""

import hashlib
import json


def canonical_counts_json(counts: dict[str, int]) -> str:
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def pool_hash(counts: dict[str, int]) -> str:
    return hashlib.sha256(canonical_counts_json(counts).encode()).hexdigest()
