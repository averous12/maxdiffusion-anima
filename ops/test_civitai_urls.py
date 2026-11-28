"""CivitAI links must survive mirrors, query strings and the token name.

Three separate faults, all hit by the same real link
https://civitai.red/api/download/models/3264980?fileId=3148583 :

1. the matcher was `civitai\\.com` only, so a .red (or any mirror) link fell through to
   "unrecognised link: expected a HuggingFace or CivitAI URL";
2. the /api/download/ branch returned `u.split("?")[0]`, dropping `fileId` -- which SELECTS the
   file, so a model version with several files fetched the wrong one even on .com;
3. the token was read from ANIMA_CIVITAI_TOKEN only, while secrets.env and the notebook form set
   CIVITAI_TOKEN, so a correctly configured session still sent no token and got 401.

The host is deliberately preserved rather than rewritten to civitai.com: a mirror is usually a
mirror because .com is unreachable from where the user is.

No network here. Tokens are never printed, only their presence.

Run from the workspace root.
"""
import importlib
import os
import sys

sys.path.insert(0, "maxdiffusion/scripts")

USER_URL = "https://civitai.red/api/download/models/3264980?fileId=3148583"


def fresh(env):
    """Import add_lora with a controlled environment (it reads tokens at import)."""
    for k in ("CIVITAI_TOKEN", "ANIMA_CIVITAI_TOKEN", "HF_TOKEN"):
        os.environ.pop(k, None)
    os.environ.update(env)
    if "add_lora" in sys.modules:
        del sys.modules["add_lora"]
    return importlib.import_module("add_lora")


def params(dl_url):
    if "?" not in dl_url:
        return []
    return [p.split("=")[0] for p in dl_url.split("?", 1)[1].split("&")]


def main():
    fails = []
    tok = {"CIVITAI_TOKEN": "test-token-value"}
    a = fresh(tok)

    # 1. the reported link: mirror host, fileId, token
    dl, name = a.resolve_url(USER_URL)
    ok = (dl.startswith("https://civitai.red/api/download/models/3264980?")
          and "fileId" in params(dl) and "token" in params(dl)
          and name.endswith(".safetensors") and "3148583" in name)
    print(f"{'PASS' if ok else 'FAIL'}  civitai.red + fileId accepted -> host="
          f"{dl.split('/')[2]} params={params(dl)} name={name}")
    if not ok:
        fails.append("red+fileId")

    # 2. .com page link still converts to the API route with the version
    dl2, _ = a.resolve_url("https://civitai.com/models/12345?modelVersionId=999")
    ok2 = ("/api/download/models/12345" in dl2 and "version" in params(dl2))
    print(f"{'PASS' if ok2 else 'FAIL'}  /models/ page link -> api route, params={params(dl2)}")
    if not ok2:
        fails.append("page link")

    # 3. fileId kept even with no token anywhere (auth is a separate concern from picking a file)
    b = fresh({})
    dl3, _ = b.resolve_url(USER_URL)
    ok3 = "fileId" in params(dl3) and "token" not in params(dl3)
    print(f"{'PASS' if ok3 else 'FAIL'}  fileId kept without a token, params={params(dl3)}")
    if not ok3:
        fails.append("no-token")

    # 4. the legacy name still works (boot.py exports ANIMA_CIVITAI_TOKEN)
    c = fresh({"ANIMA_CIVITAI_TOKEN": "legacy-value"})
    seen = c.CIVITAI_TOKEN == "legacy-value"
    dl4, _ = c.resolve_url(USER_URL)
    ok4 = seen and "token" in params(dl4)
    print(f"{'PASS' if ok4 else 'FAIL'}  ANIMA_CIVITAI_TOKEN still honoured (token={bool(seen)})")
    if not ok4:
        fails.append("legacy name")

    # 5. an unrelated link is still refused, with a message that says what is supported
    try:
        a.resolve_url("https://example.com/not-a-model")
        print("FAIL  unrelated link was accepted")
        fails.append("reject unknown")
    except ValueError as e:
        print(f"PASS  unrelated link refused: {str(e)[:60]}...")

    # 6. the file-name sanitizer does not mangle the generated name
    ok6 = a.sanitize_name("civitai_3264980_f3148583.safetensors") == \
        "civitai_3264980_f3148583.safetensors"
    print(f"{'PASS' if ok6 else 'FAIL'}  generated name survives sanitize_name")
    if not ok6:
        fails.append("sanitize")

    print()
    print("RESULT: CIVITAI URLS OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
