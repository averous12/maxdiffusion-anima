"""
Prompt-control syntax for the Anima TPU runtime.

Implements the asagi4/comfyui-prompt-control scheduling grammar plus ComfyUI
core weighting, evaluated against a step fraction so the denoise loop can switch
conditioning mid-run.

Supported
---------
Weights (ComfyUI core):
    (text:1.2)      explicit weight, nests multiplicatively
    (text)          == (text:1.1)   [weight_mode "comfy", the default]
    [text]          == (text:0.909) [weight_mode "a1111" only]
    {text}          == (text:1.05)  [weight_mode "a1111" only]
    \\:  \\#  \\\\  \\(  \\)  \\[  \\]  \\{  \\}  \\,   escapes

Schedules (prompt-control):
    [before:after:X]        switch at fraction X
    [cat:0.5]               == [:cat:0.5]
    [before:during:after:A,B]   range form
    [during:0.1,0.4]        == [:during::0.1,0.4]
    [a|b:0.1]               alternate every 0.1 (default 0.1); >2 options allowed
    [SEQ:a:0.2:b:0.5:c]     switch a -> b -> c -> nothing
    [before:after:TAG]      tag selection (uppercase tag + a `tags` request field)
    nesting is allowed in every position

Combinators (processed after schedule expansion):
    AND                 split into prompt segments (combined like CAT here)
    CAT                 encode each part separately, concatenate the tensors
    AVG(w)              encode each part, weighted-average the tensors (w default 0.5)
    BREAK               tokenise in separate chunks. NO-OP for T5 (our encoder),
                        exactly as prompt-control documents; use CAT instead
    "quoted AND"        operators inside double quotes stay literal text

Also handled: `#` comments to end of line, `STYLE(mode)` weight-mode override,
`NOSTRIP()` (accepted, whitespace already preserved here).

Deliberately out of scope: LoRA scheduling (our LoRA is merged at boot before
the sampler exists), regional prompting / MASK / COUPLE / CUT, TE() per-encoder
prompts, SDXL(), NOISE(), macros and Jinja templating.
"""

from __future__ import annotations

import hashlib
import random
import re
import numpy as np

WEIGHT_MODES = ("comfy", "a1111", "safe")
DEFAULT_EMPHASIS = {"comfy": 1.1, "a1111": 1.1, "safe": 1.1}
BRACKET_DEEMPHASIS = 1.0 / 1.1
BRACE_EMPHASIS = 1.05

# Operators recognised in literal text at word boundaries.
_OP_AND = "AND"
_OP_CAT = "CAT"
_OP_BREAK = "BREAK"
_AVG_RE = re.compile(r"^AVG\(([-+]?\d*\.?\d*)\)$")


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #
# ("text", str)
# ("nodes", [node, ...])                      concatenation
# ("weight", node, float)                     explicit or implicit emphasis
# ("sched", before_nodes, after_nodes, x)     [a:b:x]
# ("range", before, during, after, a, b)      [a:b:c:A,B]
# ("alt", [nodes, ...], pct)                  [a|b:pct]
# ("seqstep", [(nodes, at), ...])             [SEQ:...]
# ("tagsel", before, after, tag)              [a:b:TAG]
# ("choice", [nodes, ...])                    {a|b|c}


class PromptSyntaxError(ValueError):
    pass


def _strip_comments(s: str) -> str:
    """Remove `#` comments to end of line, honouring escapes and quotes."""
    out, i, in_q = [], 0, False
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(c); out.append(s[i + 1]); i += 2; continue
        if c == '"':
            in_q = not in_q
            out.append(c); i += 1; continue
        if c == "#" and not in_q:
            while i < len(s) and s[i] != "\n":
                i += 1
            continue
        out.append(c); i += 1
    return "".join(out)


def _find_match(s: str, start: int, open_c: str, close_c: str, lbracket=False):
    """Index of the matching close for the opener at `start`, or -1."""
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2; continue
        if c == '"':
            i += 1
            while i < len(s) and s[i] != '"':
                if s[i] == "\\":
                    i += 1
                i += 1
            i += 1; continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_top(s: str, sep: str):
    """Split on `sep` at nesting depth 0, honouring escapes and quotes."""
    parts, buf, depth, i, in_q = [], [], 0, 0, False
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            buf.append(c); buf.append(s[i + 1]); i += 2; continue
        if c == '"':
            in_q = not in_q
            buf.append(c); i += 1; continue
        if not in_q:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == sep and depth == 0:
                parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    parts.append("".join(buf))
    return parts


def _is_number(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_tag(s: str) -> bool:
    s = s.strip()
    return bool(s) and re.fullmatch(r"[A-Z_]+", s) is not None


def _parse_nodes(s: str, mode: str) -> list:
    """Parse a string into a list of AST nodes (no AND/CAT handling here)."""
    nodes, buf, i = [], [], 0

    def flush():
        if buf:
            nodes.append(("text", "".join(buf)))
            buf.clear()

    while i < len(s):
        c = s[i]

        if c == "\\" and i + 1 < len(s):
            buf.append(c); buf.append(s[i + 1]); i += 2; continue

        # Operators are recognised at PARSE time, so parentheses that belong to
        # AVG(...) are consumed as part of the operator instead of being mistaken
        # for a weight group. Quoted spans are copied wholesale below and never
        # reach this branch, which is exactly the required behaviour.
        m = _OP_RE.match(s, i)
        if m:
            flush()
            tok = m.group(1)
            am = _AVG_RE.match(tok)
            if am:
                arg = am.group(1)
                nodes.append(("op", "AVG",
                              float(arg) if arg not in ("", ".") else 0.5))
            else:
                nodes.append(("op", tok, None))
            i = m.end(); continue

        if c == '"':
            j = i + 1
            lit = ['"']
            while j < len(s) and s[j] != '"':
                if s[j] == "\\":
                    lit.append(s[j]); j += 1
                    if j < len(s):
                        lit.append(s[j])
                else:
                    lit.append(s[j])
                j += 1
            if j < len(s):
                lit.append('"'); j += 1
            buf.extend(lit); i = j; continue

        if c == "(":
            j = _find_match(s, i, "(", ")")
            if j != -1:
                inner = s[i + 1:j]
                parts = _split_top(inner, ":")
                w = None
                if len(parts) > 1 and _is_number(parts[-1]):
                    w = float(parts[-1].strip())
                    inner = ":".join(parts[:-1])
                if w is None and mode == "safe":
                    # "safe" keeps plain parentheses literal, so danbooru-style
                    # tags like "fern (sousou no frieren)" survive untouched.
                    # Only the explicit "(...:number)" form is a weight.
                    flush()
                    nodes.append(("text", "("))
                    nodes.extend(_parse_nodes(inner, mode))
                    nodes.append(("text", ")"))
                    i = j + 1; continue
                flush()
                if w is None:
                    nodes.append(("weight", ("nodes", _parse_nodes(inner, mode)),
                                  DEFAULT_EMPHASIS[mode]))
                else:
                    nodes.append(("weight", ("nodes", _parse_nodes(inner, mode)), w))
                i = j + 1; continue

        if c == "[":
            j = _find_match(s, i, "[", "]")
            if j != -1:
                inner = s[i + 1:j]
                flush()
                if (mode == "a1111" and ":" not in inner and "|" not in inner
                        and not inner.strip().upper().startswith("SEQ")):
                    # A1111 style: [x] is de-emphasis, not a schedule.
                    nodes.append(("weight", ("nodes", _parse_nodes(inner, mode)),
                                  BRACKET_DEEMPHASIS))
                else:
                    nodes.append(_parse_schedule(inner, mode))
                i = j + 1; continue

        if c == "{":
            j = _find_match(s, i, "{", "}")
            if j != -1:
                inner = s[i + 1:j]
                flush()
                if "|" in inner:
                    opts = [_parse_nodes(p, mode) for p in _split_top(inner, "|")]
                    nodes.append(("choice", opts))
                elif mode == "a1111":
                    nodes.append(("weight", ("nodes", _parse_nodes(inner, mode)),
                                  BRACE_EMPHASIS))
                else:
                    buf.append(c); i += 1; continue
                i = j + 1; continue

        buf.append(c); i += 1

    flush()
    return nodes


def _parse_schedule(inner: str, mode: str):
    """Parse the body of a [...] expression."""
    parts = _split_top(inner, ":")

    # [SEQ:a:N1:b:N2:c]
    if parts and parts[0].strip().upper() == "SEQ":
        items = []
        rest = parts[1:]
        k = 0
        while k < len(rest):
            label = rest[k]
            at = None
            if k + 1 < len(rest) and _is_number(rest[k + 1]):
                at = float(rest[k + 1].strip())
                k += 1
            items.append((_parse_nodes(label, mode), at))
            k += 1
        return ("seqstep", items)

    # [a|b:pct]  alternating
    if any("|" in p for p in parts):
        pct = 0.1
        if _is_number(parts[-1]) and len(parts) > 1:
            pct = float(parts[-1].strip())
            parts = parts[:-1]
        opts = []
        for p in parts:
            opts.extend([_parse_nodes(x, mode) for x in _split_top(p, "|")])
        return ("alt", opts, pct)

    # range forms: last part holds "A,B"
    if len(parts) >= 2 and "," in parts[-1] and len(parts[-1].split(",")) == 2:
        lo, hi = [x.strip() for x in parts[-1].split(",")]
        if _is_number(lo) and _is_number(hi):
            body = parts[:-1]
            if len(body) == 1:                      # [during:0.1,0.4]
                before, during, after = "", body[0], ""
            elif len(body) == 2:                    # [during:after:0.1,0.4]
                before, during, after = "", body[0], body[1]
            else:                                   # [before:during:after:A,B]
                before, during, after = body[-3], body[-2], body[-1]
            return ("range",
                    ("nodes", _parse_nodes(before, mode)),
                    ("nodes", _parse_nodes(during, mode)),
                    ("nodes", _parse_nodes(after, mode)), float(lo), float(hi))

    if len(parts) == 1:
        # [cat:0.5] handled below as 2-part; a bare [...] is just text
        return ("nodes", _parse_nodes(parts[0], mode))

    if len(parts) == 2:
        a, b = parts
        if _is_number(b):                           # [cat:0.5] == [:cat:0.5]
            return ("sched", ("nodes", []), ("nodes", _parse_nodes(a, mode)),
                    float(b.strip()))
        return ("nodes", _parse_nodes(inner, mode))

    if len(parts) == 3:
        a, b, c = parts
        if _is_number(c):
            return ("sched", ("nodes", _parse_nodes(a, mode)),
                    ("nodes", _parse_nodes(b, mode)), float(c.strip()))
        if _is_tag(c):
            return ("tagsel", ("nodes", _parse_nodes(a, mode)),
                    ("nodes", _parse_nodes(b, mode)), c.strip().upper())
        return ("nodes", _parse_nodes(inner, mode))

    if len(parts) == 4:
        a, b, c, d = parts
        if _is_tag(d):
            return ("tagsel", ("nodes", _parse_nodes(a, mode)),
                    ("nodes", _parse_nodes(c, mode)), d.strip().upper())
        return ("nodes", _parse_nodes(inner, mode))

    return ("nodes", _parse_nodes(inner, mode))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _append_text(seg, text):
    if seg and seg[-1][0] == "text":
        seg[-1] = ("text", seg[-1][1] + text)
    else:
        seg.append(("text", text))


def _eval(nodes, frac, tags, seg, rng, mode):
    """Evaluate a LIST of AST nodes at `frac`."""
    for node in nodes:
        _eval_one(node, frac, tags, seg, rng, mode)


def _eval_one(node, frac, tags, seg, rng, mode):
    """Evaluate a SINGLE AST node.

    Note the two shapes: a node list is a list of tuples, while a grouped
    sub-expression is the tuple ("nodes", [node, ...]). Iterating that tuple
    directly would unpack the string "nodes" into per-character kinds.
    """
    kind, *rest = node
    if kind == "text":
        _append_text(seg, rest[0])
    elif kind == "op":
        seg.append(("op", rest[0], rest[1]))
    elif kind == "nodes":
        _eval(rest[0], frac, tags, seg, rng, mode)
    elif kind == "weight":
        inner, w = rest
        sub = []
        _eval_one(inner, frac, tags, sub, rng, mode)
        seg.extend(_scale(sub, w))
    elif kind == "choice":
        _eval(rng.choice(rest[0]), frac, tags, seg, rng, mode)
    elif kind == "sched":
        before, after, x = rest
        _eval_one(before if frac < x else after, frac, tags, seg, rng, mode)
    elif kind == "range":
        before, during, after, a, b = rest
        pick = before if frac < a else (during if frac < b else after)
        _eval_one(pick, frac, tags, seg, rng, mode)
    elif kind == "alt":
        opts, pct = rest
        pct = pct if pct > 0 else 0.1
        _eval(opts[int(frac / pct) % len(opts)], frac, tags, seg, rng, mode)
    elif kind == "seqstep":
        chosen = None
        for label, at in rest[0]:
            if at is None or frac < at:
                chosen = label
                break
        if chosen is None:
            chosen = []
        _eval(chosen, frac, tags, seg, rng, mode)
    elif kind == "tagsel":
        before, after, tag = rest
        hit = bool(tags) and tag in {str(t).upper() for t in tags}
        _eval_one(after if hit else before, frac, tags, seg, rng, mode)
    else:
        raise PromptSyntaxError(f"unknown node {kind!r}")


def _scale(seg, w):
    """Multiply every (text, weight) pair in `seg` by w (weights applied later)."""
    out = []
    for item in seg:
        if item[0] == "text":
            out.append(("wtext", item[1], w))
        elif item[0] == "wtext":
            out.append(("wtext", item[1], item[2] * w))
        else:
            out.append(item)
    return out


def _flatten_weighted(seg):
    """Collapse to [("text", s) | ("op", name, arg) | ("wtext", s, w)]."""
    merged = []
    for item in seg:
        if item[0] == "text":
            if merged and merged[-1][0] in ("text", "wtext"):
                prev = merged[-1]
                if prev[0] == "text":
                    merged[-1] = ("text", prev[1] + item[1])
                else:
                    merged[-1] = ("wtext", prev[1] + item[1], prev[2])
            else:
                merged.append(item)
        else:
            merged.append(item)
    return merged


# Word operators. `\b` is wrong here: `)` is a non-word char, so `AVG(0.5) x`
# has no word boundary after the paren. Use explicit neighbour lookarounds.
_OP_RE = re.compile(r"(?<![A-Za-z0-9_])(AND|CAT|BREAK|AVG\([^)]*\))(?![A-Za-z0-9_])")


def _terms(seg):
    """Split the evaluated stream into terms joined by AND / CAT / AVG.

    Ops describe how to reduce the terms left to right:
        p1 AND p2          -> terms [p1, p2], ops ["AND"]
        p1 CAT p2          -> terms [p1, p2], ops ["CAT"]
        p1 AVG(0.3) p2     -> terms [p1, p2], ops [("AVG", 0.3)]
    BREAK is dropped here: prompt-control documents it as a no-op for T5.
    """
    terms = [[]]
    ops = []
    for item in seg:
        if item[0] == "op":
            name, arg = item[1], item[2]
            if name == _OP_BREAK:
                continue
            ops.append(("AVG", float(arg)) if name == "AVG" else (name, None))
            terms.append([])
        else:
            terms[-1].append(item)

    # drop empty terms (e.g. a leading/trailing operator) and their op
    i = 0
    while i < len(terms):
        if not terms[i]:
            terms.pop(i)
            if i < len(ops):
                ops.pop(i)
            elif ops:
                ops.pop()
            continue
        i += 1

    if not terms:
        terms = [[]]
    return terms, ops


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
_STYLE_RE = re.compile(r"(?<![A-Za-z0-9_])STYLE\(\s*([a-zA-Z0-9_+]+)\s*\)")
_NOSTRIP_RE = re.compile(r"(?<![A-Za-z0-9_])NOSTRIP\(\s*\)")


def preprocess(text, weight_mode="comfy"):
    """Handle STYLE(name) / NOSTRIP() and return (cleaned_text, weight_mode).

    STYLE selects the weight interpretation for the whole prompt. prompt-control
    allows it per AND-segment; we apply it globally, which is the common case.
    NOSTRIP() is accepted and removed - whitespace is already preserved here.
    """
    text = text or ""
    mode = weight_mode
    m = _STYLE_RE.search(text)
    if m:
        name = m.group(1).strip().lower()
        if name in WEIGHT_MODES:
            mode = name
        text = _STYLE_RE.sub("", text)
    text = _NOSTRIP_RE.sub("", text)
    return text, mode


def parse(text, weight_mode="comfy"):
    text, mode = preprocess(text, weight_mode)
    return _parse_nodes(_strip_comments(text), mode)


def evaluate(ast, frac, tags=None, seed=0, weight_mode="comfy"):
    """Evaluate the AST at a step fraction.

    Returns {"terms": [term, ...], "ops": [(name, arg), ...]} where term i and
    term i+1 are joined by ops[i]. Each term is a list of ("text"|"wtext", ...)
    items; use `terms_to_text` to get readable (text, char_weights) pairs.
    """
    rng = random.Random(f"{seed}|{text_key(ast)}")
    seg = []
    _eval(ast, float(frac), tags, seg, rng, weight_mode)
    terms, ops = _terms(_flatten_weighted(seg))
    return {"terms": terms, "ops": ops}


def terms_to_text(result, collapse_ws=True):
    """-> (list of (text, char_weights), list of ops) for an evaluate() result."""
    texts = [_segments_to_text(t, collapse_ws=collapse_ws) for t in result["terms"]]
    return texts, list(result["ops"])


def text_key(ast):
    return hashlib.sha1(repr(ast).encode()).hexdigest()[:12]


def _segments_to_text(term, collapse_ws=True):
    """-> (clean_text, per-char weights) for one term.

    Whitespace runs are collapsed to a single space and the ends are stripped,
    matching prompt-control's default stripping. The weight array is rebuilt in
    lockstep so it stays aligned with the text.
    """
    parts = []
    for item in term:
        s = item[1]
        w = 1.0 if item[0] == "text" else item[2]
        parts.append((s, w))
    if not parts:
        return "", np.zeros(0, np.float32)

    text = "".join(s for s, _ in parts)
    wts = np.concatenate([np.full(len(s), w, np.float32) for s, w in parts])
    if not collapse_ws:
        return text, np.clip(wts, 0.0, 3.0)

    chars, ws = [], []
    prev_ws = True                     # also strips leading whitespace
    for ch, w in zip(text, wts):
        if ch.isspace():
            if prev_ws:
                continue
            chars.append(" "); ws.append(w); prev_ws = True
        else:
            chars.append(ch); ws.append(w); prev_ws = False
    while chars and chars[-1] == " ":   # strip trailing space
        chars.pop(); ws.pop()
    out = np.asarray(ws, np.float32) if ws else np.zeros(0, np.float32)
    return "".join(chars), np.clip(out, 0.0, 3.0)


def apply_escapes(s):
    """Turn the parser's kept backslash escapes into literal characters."""
    out, i = [], 0
    esc = set(":#\\()[]{},")
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in esc:
            out.append(s[i + 1]); i += 2; continue
        out.append(s[i]); i += 1
    return "".join(out)


def breakpoints(ast):
    """Step fractions where the evaluated prompt can change."""
    pts = {0.0, 1.0}

    def _one(node):
        kind, *rest = node
        if kind == "nodes":
            walk(rest[0])
        elif kind == "weight":
            _one(rest[0])
        elif kind == "choice":
            for o in rest[0]:
                walk(o)
        elif kind == "sched":
            before, after, x = rest
            pts.add(min(max(float(x), 0.0), 1.0))
            _one(before); _one(after)
        elif kind == "range":
            before, during, after, a, b = rest
            pts.add(min(max(float(a), 0.0), 1.0))
            pts.add(min(max(float(b), 0.0), 1.0))
            _one(before); _one(during); _one(after)
        elif kind == "alt":
            opts, pct = rest
            pct = pct if pct > 0 else 0.1
            k = 1
            while k * pct < 1.0:
                pts.add(k * pct); k += 1
            for o in opts:
                walk(o)
        elif kind == "seqstep":
            for label, at in rest[0]:
                if at is not None:
                    pts.add(min(max(float(at), 0.0), 1.0))
                walk(label)
        elif kind == "tagsel":
            before, after, _tag = rest
            _one(before); _one(after)

    def walk(nodes):
        for node in nodes:
            _one(node)

    walk(ast)
    return sorted(pts)


def plan(text, steps, weight_mode="comfy", tags=None, seed=0):
    """Precompute the distinct prompt states a schedule passes through.

    Returns dict with:
      states      : list of group lists (one per distinct state)
      step_state  : [state index per denoise step]
      breakpoints : the fractions where the prompt changes
      variants    : number of distinct states
    """
    ast = parse(text, weight_mode)
    bps = breakpoints(ast)
    steps = max(1, int(steps))

    # a state is chosen per step: frac = (i + 1) / steps
    fracs = [(i + 1) / steps for i in range(steps)]
    seen, states, step_state = {}, [], []
    for f in fracs:
        groups = evaluate(ast, f, tags=tags, seed=seed, weight_mode=weight_mode)
        key = repr(groups)
        if key not in seen:
            seen[key] = len(states)
            states.append(groups)
        step_state.append(seen[key])

    return {"states": states, "step_state": step_state, "breakpoints": bps,
            "variants": len(states), "ast": ast, "weight_mode": weight_mode}


def describe(planned):
    """One-line summary for logs."""
    return (f"{planned['variants']} prompt state(s), "
            f"switch points {[round(b, 3) for b in planned['breakpoints'] if 0 < b < 1]}, "
            f"weight mode {planned['weight_mode']}")
