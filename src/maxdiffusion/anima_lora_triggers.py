"""Recover a LoRA's trained trigger words from its own safetensors metadata.

Why this exists
---------------
Anima LoRAs bind their concept to a trigger token (`hsin (wuthering waves)`,
`extpose`). The server strips `<lora:name:scale>` from the prompt before the
text encoder sees it, so the tag never reaches the encoder — by design. That
means a LoRA applied without its trigger word changes the weights but never
asks the model for the concept: the image drifts, the character does not
appear. This module answers "what should be appended to the prompt?" without
a network call, using the `ss_tag_frequency` block that kohya/sd-scripts
writes into every trained file.

Heuristic
---------
Per training subset, take the tags present in *every* caption of that subset
(count == the subset's image count), union those across subsets, and drop
generic booru tags that say nothing about identity. What remains is the
concept token(s). Verified against `hsin_anima_v4.safetensors`, whose five
subsets yield exactly its two advertised trigger forms.
"""
import json
import os
import struct

# Tags that appear in every caption of a character dataset but carry no
# concept: framing, count, render style, background.
GENERIC_TAGS = {
    "1girl", "1boy", "2girls", "solo", "solo focus", "multiple girls",
    "looking at viewer", "looking back", "from side", "from behind",
    "from above", "straight-on", "profile", "portrait", "close-up",
    "upper body", "lower body", "full body", "cowboy shot", "head out of frame",
    "eyes out of frame", "feet out of frame", "foot out of frame",
    "3d", "render", "tachi-e", "realistic", "photo", "anime",
    "simple background", "white background", "grey background",
    "black background", "gradient background", "blue background",
    "red background", "brown background", "blurry background",
    "detailed background", "indoors", "outdoors", "day", "night",
    "highres", "absurdres", "masterpiece", "best quality", "very aesthetic",
    "breasts", "cleavage", "bare shoulders", "collarbone", "thighs",
    "closed mouth", "open mouth", "parted lips", "smile", "blush",
    "standing", "sitting", "lying", "arms up", "hand up",
}


def read_metadata(path):
    """The `__metadata__` block of a safetensors file, without loading tensors."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            return {}
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > 100 * 1024 * 1024:
            return {}
        try:
            header = json.loads(f.read(n))
        except Exception:
            return {}
    return header.get("__metadata__", {}) or {}


def trained_words(path, limit=4):
    """Trigger words for `path`, most concept-like first. [] when unknown."""
    meta = read_metadata(path)
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return []
    try:
        freq = json.loads(raw)
    except Exception:
        return []
    if not isinstance(freq, dict) or not freq:
        return []

    per_subset = []
    for _subset, tags in freq.items():
        if not isinstance(tags, dict) or not tags:
            continue
        total = max(int(c) for c in tags.values())
        if total <= 0:
            continue
        per_subset.append({t for t, c in tags.items() if int(c) >= total})
    if not per_subset:
        return []

    candidates = set()
    coverage = {}
    for s in per_subset:
        candidates |= s
        for t in s:
            coverage[t] = coverage.get(t, 0) + 1
    candidates = {t for t in candidates if t.strip().lower() not in GENERIC_TAGS}

    def score(tag):
        # Coverage first: a trigger used by every subset beats a variant that
        # only one subset mentions. Then prefer the shortest form, because the
        # base trigger ("hsin (wuthering waves)") is the general-purpose one and
        # the longer variants ("hsin (illumining form) (wuthering waves)") are
        # specialisations the user should opt into.
        return (coverage.get(tag, 0), -len(tag))

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[:limit]


def triggers_for(specs):
    """{basename: [words]} for [(path, scale), ...] or [path, ...]."""
    out = {}
    for item in specs or []:
        path = item[0] if isinstance(item, (tuple, list)) else item
        if not path:
            continue
        out[os.path.basename(str(path))] = trained_words(path)
    return out


def missing_triggers(prompt, words):
    """Trigger words from `words` that `prompt` does not already contain."""
    low = (prompt or "").lower()
    return [w for w in (words or []) if w.lower() not in low]
