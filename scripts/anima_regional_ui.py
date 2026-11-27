"""Regional prompt editor widgets and request assembly for the Gradio UI.

Kept out of anima_gradio_ui.py so the widget layout and the request assembly can be
tested without building the whole app. The JSON stays the wire format: the editor is a
view over it, and a hand-written JSON box still overrides the editor when filled.
"""
from __future__ import annotations

import base64

FIELDS = ("enabled", "prompt", "negative", "x", "y", "w", "h",
          "start", "end", "mode", "ref1", "ref2", "ref3", "weights")

DEFAULT_MP = 0.25


def build_slots(count: int = 4):
    """Create the region widgets. Call inside the Blocks context."""
    import gradio as gr

    slots = []
    for i in range(count):
        with gr.Accordion(f"Box {i + 1}", open=(i == 0)):
            with gr.Row():
                # Off by default. It used to be value=(i == 0), which silently enabled one
                # box covering the left half (x=0, w=0.5) with a blank prompt -- so every
                # untouched generation was routed through the regional path and nobody had
                # asked for it. Regional is now opt-in: tick Include.
                enabled = gr.Checkbox(value=False, label="Include", scale=0)
                mode = gr.Dropdown(["char", "style", "cs"], value="char",
                                   label="Ref mode", scale=0)
                start = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Start")
                end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="End")
            prompt = gr.Textbox(lines=2, label="Box prompt",
                                placeholder="1girl, solo, red raincoat, night city background",
                                info="The common block above is prepended to this.")
            negative = gr.Textbox(lines=1, label="Box negative (optional)")
            with gr.Row():
                x = gr.Number(value=0.0, label="x", minimum=0.0, maximum=1.0, step=0.01)
                y = gr.Number(value=0.0, label="y", minimum=0.0, maximum=1.0, step=0.01)
                w = gr.Number(value=0.5, label="w", minimum=0.05, maximum=1.0, step=0.01)
                h = gr.Number(value=1.0, label="h", minimum=0.05, maximum=1.0, step=0.01)
            with gr.Row():
                ref1 = gr.Image(type="filepath", label="Ref 1", height=90)
                ref2 = gr.Image(type="filepath", label="Ref 2", height=90)
                ref3 = gr.Image(type="filepath", label="Ref 3", height=90)
            weights = gr.Textbox(value="", label="Ref weights",
                                 placeholder="1.0, 0.8 — blank means all 1.0")
        slots.append({name: comp for name, comp in zip(FIELDS, (
            enabled, prompt, negative, x, y, w, h, start, end, mode,
            ref1, ref2, ref3, weights))})
    return slots


def ordered_inputs(slots):
    """Flat component list in FIELDS order, ready for a Gradio `inputs=`."""
    out = []
    for slot in slots:
        out.extend(slot[name] for name in FIELDS)
    return out


def _weights(text, count):
    vals = []
    for chunk in str(text or "").replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            vals.append(float(chunk))
        except ValueError:
            pass
    vals = (vals + [1.0] * count)[:count]
    return vals or [1.0] * count


def _b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _box(d):
    def num(key, default):
        try:
            return float(d.get(key))
        except (TypeError, ValueError):
            return default
    x, y = num("x", 0.0), num("y", 0.0)
    w, h = num("w", 0.5), num("h", 1.0)
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.05), 1.0 - x) if x < 1.0 else 0.05
    h = min(max(h, 0.05), 1.0 - y) if y < 1.0 else 0.05
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def assemble(values, include_refs: bool = True):
    """Turn the flat widget values into the request's region list.

    Unchecked boxes are skipped entirely, which is what makes the checkmark free:
    the server never sees them, so they cost no encode and no branch.
    """
    step = len(FIELDS)
    regions = []
    for index in range(len(values) // step):
        d = dict(zip(FIELDS, values[index * step:(index + 1) * step]))
        if not d.get("enabled"):
            continue
        region = {
            "prompt": (d.get("prompt") or "").strip(),
            "box": _box(d),
            "start": float(d.get("start") or 0.0),
            "end": float(d.get("end") or 1.0),
            "enabled": True,
            # 1-based widget number, so a validation message can name the box the user
            # actually ticked. Ignored by the server's canonicaliser.
            "slot": index + 1,
        }
        negative = (d.get("negative") or "").strip()
        if negative:
            region["negative_prompt"] = negative
        images = [p for p in (d.get("ref1"), d.get("ref2"), d.get("ref3")) if p]
        if images:
            region["snapref"] = {
                "enabled": True,
                "mode": d.get("mode") or "char",
                "ref_megapixels": DEFAULT_MP,
                "ref_mp_budget": round(DEFAULT_MP * len(images), 2),
                "weights": _weights(d.get("weights"), len(images)),
            }
            if include_refs:
                region["ref_images_b64"] = [_b64(p) for p in images]
        regions.append(region)
    return regions


def _coverage(regions, canvas_w: int = 1024, canvas_h: int = 1024):
    """Share of the canvas short of, or over, mask mass 1.

    The grid follows the canvas aspect so a cell covers equal area in both axes;
    a square grid on a wide canvas would mis-weight the percentage.
    """
    cols = 32
    rows = max(1, round(cols * float(canvas_h) / max(1.0, float(canvas_w))))
    total = [[0.0] * cols for _ in range(rows)]
    for region in regions:
        x, y, w, h = region["box"]
        for gy in range(rows):
            cy = (gy + 0.5) / rows
            if not (y <= cy < y + h):
                continue
            for gx in range(cols):
                cx = (gx + 0.5) / cols
                if x <= cx < x + w:
                    total[gy][gx] += 1.0
    cells = [v for row in total for v in row]
    gaps = sum(1 for v in cells if v < 0.999)
    overlap = sum(1 for v in cells if v > 1.001)
    return gaps / len(cells), overlap / len(cells)


def preview_html(values, common_positive: str = "", common_negative: str = "",
                 canvas_w: int = 1024, canvas_h: int = 1024):
    """Server-rendered canvas preview with a coverage verdict.

    The preview box follows the generation's aspect ratio, so a 16:9 run shows a wide
    canvas and a half-width box really is half the width. Shows where the base prompt
    leaks through (gaps) and where regions dilute each other (overlap) — the two
    failure modes that are invisible in a JSON box. Colours are literal: this HTML is
    rendered by Gradio, which does not define the host theme's CSS variables.
    """
    regions = assemble(values, include_refs=False)
    # Theme-neutral greys: a white border is invisible on Gradio's light theme, which
    # made the canvas boundary unreadable and hid the aspect change.
    border = "rgba(128,128,128,.55)"
    muted = "rgba(140,140,140,1)"

    # Longest side is fixed so the panel never jumps around between aspects.
    long_side = 200
    canvas_w = max(1, int(canvas_w or 1024))
    canvas_h = max(1, int(canvas_h or 1024))
    scale = long_side / max(canvas_w, canvas_h)
    box_w = max(40, round(canvas_w * scale))
    box_h = max(40, round(canvas_h * scale))

    boxes = []
    for i, region in enumerate(regions):
        x, y, w, h = region["box"]
        boxes.append(
            f'<div style="position:absolute;left:{x*100:.2f}%;top:{y*100:.2f}%;'
            f'width:{w*100:.2f}%;height:{h*100:.2f}%;border:2px solid '
            f'{"#e8734a #4a9de8 #9a7ae0 #3fb98a".split()[i % 4]};'
            f'border-radius:4px;font-size:10px;">'
            f'<span style="background:rgba(0,0,0,.62);color:#fff;padding:0 3px;'
            f'border-radius:0 0 3px 0;">R{i + 1}</span></div>')

    gap, overlap = _coverage(regions, canvas_w, canvas_h) if regions else (0.0, 0.0)
    if not regions:
        verdict = "No boxes checked — runs as a plain generation."
        colour = muted
    elif gap > 0.001:
        verdict = (f"{gap*100:.0f}% of the canvas is uncovered — the base prompt "
                   f"draws there. That is how the style drifts.")
        colour = "#d08a2a"
    elif overlap > 0.001:
        verdict = (f"{overlap*100:.0f}% is covered twice — overlapping boxes dilute "
                   f"each other against the same base.")
        colour = "#d08a2a"
    else:
        verdict = "Coverage complete — the base prompt is fully replaced inside regions."
        colour = "#3fb98a"

    common_note = ""
    if not (common_positive or "").strip():
        common_note = ("<div style='color:#d08a2a;font-size:11px;margin-top:4px;'>"
                       "Common prompt is empty: each box must carry its own quality "
                       "preamble and artist tags.</div>")
    if regions and not any("snapref" in r for r in regions):
        refs = "no box has a reference"
    else:
        refs = f"{sum(1 for r in regions if 'snapref' in r)} box(es) with references"

    return (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div id="regional-canvas" style="position:relative;width:{box_w}px;height:{box_h}px;'
        f'border:1px solid {border};border-radius:8px;overflow:hidden;flex:0 0 auto;'
        f'background:repeating-conic-gradient(from 0deg,'
        f'rgba(128,128,128,.13) 0deg 90deg, transparent 90deg 180deg) 0 0 / 25px 25px;">'
        + "".join(boxes) +
        '</div>'
        f'<div style="flex:1 1 200px;min-width:180px;font-size:11.5px;">'
        f'<div style="color:{muted};margin-bottom:4px;">{canvas_w}×{canvas_h}</div>'
        f'<div style="color:{colour};margin-bottom:6px;">{verdict}</div>'
        f'<div style="color:{muted};">{len(regions)} box(es) active'
        f' · {2 + 2 * len(regions)} model calls/step vs 2 plain · {refs}</div>'
        f'{common_note}'
        '</div></div>'
    )
