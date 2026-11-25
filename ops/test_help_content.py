"""Guards the Help tab: every section is actually rendered, and the docs stay true.

The prompt-syntax help existed as a constant for a long time but was never placed in
any tab, so nobody could read it. These checks fail if that happens again, or if the
docs start contradicting the features.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ.setdefault("ANIMA_CONTENT_ROOT",
                      os.path.join(os.environ.get("TEMP", "/tmp"), "ui_help_probe"))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import anima_gradio_ui as ui  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


def main():
    config = ui.build().config
    docs = " ".join(c.get("props", {}).get("value") or ""
                    for c in config["components"] if c.get("type") == "markdown")

    for name, text in (("tabs", ui.TABS_HELP), ("regional", ui.REGIONAL_HELP),
                       ("perf", ui.PERF_HELP), ("api", ui.API_HELP),
                       ("trouble", ui.TROUBLE_HELP), ("syntax", ui.SYNTAX_HELP)):
        check(f"{name} help is rendered in a tab", text[:60] in docs,
              "constant defined but never placed in the UI")

    accordions = [(c.get("props") or {}).get("label") for c in config["components"]
                  if c.get("type") == "accordion"]
    for section in ("What each tab does", "Prompt syntax", "Regional prompting",
                    "Performance and cost", "API and CLI", "Troubleshooting"):
        check(f"section {section!r}", section in accordions)

    # The docs must not claim a feature is missing when we ship it.
    check("docs no longer deny regional prompting",
          "Not supported:** regional prompting" not in ui.SYNTAX_HELP)
    check("docs mention the common block", "Common prompt" in ui.REGIONAL_HELP)
    check("docs mention per-box references", "Ref 1" in ui.REGIONAL_HELP)
    check("docs state the base prompt is replaced",
          "contributes nothing" in ui.REGIONAL_HELP)
    check("docs give the cost model", "2 + 2N" in ui.PERF_HELP)
    check("docs cover regional_end", "regional_end" in ui.API_HELP)

    # Every non-structural control label should be findable in the docs.
    # Labels carry inline hints ("Seed (-1 = random)"), so compare the base name.
    skip = {"tabitem", "markdown", "html", "button", "accordion"}
    missing = []
    for comp in config["components"]:
        if comp.get("type") in skip:
            continue
        label = ((comp.get("props") or {}).get("label") or "").strip()
        if not label or label == "_":
            continue
        base = label.split("(")[0].strip().rstrip("—").strip()
        if base in docs:
            continue
        missing.append(base)
    check("every control is documented", not missing, f"undocumented: {sorted(set(missing))}")

    print(f"HELP_CONTENT {PASS}/{PASS + FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
