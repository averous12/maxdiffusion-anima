"""Which control features are active for a request, and which combinations are allowed.

Pure by design: no JAX, no model, no I/O. It lives outside the server so the rules can be
tested without booting the runtime, and so the server and any other caller cannot drift
apart on what "lllite + snapref" means.

The six supported combinations come from the shipped node:
plain, regional, snapref, lllite, regional+snapref, lllite+snapref.
"""
from __future__ import annotations

from typing import Any, Dict

# Exactly the combinations the generate path supports, in the order we report them.
SUPPORTED_CONTROL_COMBOS = (
    "plain",
    "regional",
    "snapref",
    "lllite",
    "regional+snapref",
    "lllite+snapref",
)

# canonical name per active-feature set; order-independent so callers need not care
# whether regional was declared before or after snapref
_CANONICAL = {
    frozenset(): "plain",
    frozenset({"regional"}): "regional",
    frozenset({"snapref"}): "snapref",
    frozenset({"lllite"}): "lllite",
    frozenset({"regional", "snapref"}): "regional+snapref",
    frozenset({"lllite", "snapref"}): "lllite+snapref",
}


class ControlModeError(ValueError):
    """Raised for a control combination the generate path cannot honour."""


def _snapref_active(request: Dict[str, Any]) -> bool:
    snap = request.get("snapref") or {}
    if not isinstance(snap, dict) or not snap.get("enabled", True):
        return False
    images = snap.get("images") or []
    return bool(images)


def _lllite_active(request: Dict[str, Any]) -> bool:
    lite = request.get("lllite") or {}
    if not isinstance(lite, dict) or not lite.get("enabled", True):
        return False
    # a control map is mandatory: strength alone cannot steer anything
    return lite.get("image") is not None


def _regional_active(request: Dict[str, Any]) -> bool:
    specs = request.get("regional") or request.get("regions") or []
    return bool(specs)


def resolve_control_modes(request: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a generation request into one of the six modes.

    Returns {"mode", "regional", "snapref", "lllite", "supported"} where `mode` is the
    canonical name and the booleans say which features are live.

    Raises ControlModeError for any combination absent from SUPPORTED_CONTROL_COMBOS --
    currently `regional+lllite` (both are whole-image LLLite patches, and regional replaces
    the base prediction per region, so the control would be dropped silently) and any
    three-way request.
    """
    if not isinstance(request, dict):
        raise ControlModeError(f"request must be a dict, got {type(request).__name__}")

    active = set()
    if _regional_active(request):
        active.add("regional")
    if _snapref_active(request):
        active.add("snapref")
    if _lllite_active(request):
        active.add("lllite")

    mode = _CANONICAL.get(frozenset(active))
    if mode is None:
        shown = "+".join(sorted(active)) or "plain"
        reason = ""
        if active == {"regional", "lllite"}:
            reason = (" -- regional prediction replaces the base prediction per region, so a "
                      "whole-image LLLite control would be dropped rather than applied")
        raise ControlModeError(
            f"unsupported control combination: {shown}{reason}. "
            f"Supported: {', '.join(SUPPORTED_CONTROL_COMBOS)}")

    return {
        "mode": mode,
        "regional": "regional" in active,
        "snapref": "snapref" in active,
        "lllite": "lllite" in active,
        "supported": True,
    }
