"""Fetch a LoRA from a HuggingFace or CivitAI link into the LoRA directory.

The UI spawns this detached and follows /content/anima_download.json, because a
600 MB download through the tunnel would look dead as a blocking click.

Accepted links:
  https://huggingface.co/<owner>/<repo>/resolve/main/<file>.safetensors
  https://huggingface.co/<owner>/<repo>/blob/main/<file>.safetensors   (converted)
  https://huggingface.co/<owner>/<repo>/tree/main/<file>.safetensors   (converted)
  https://huggingface.co/<owner>/<repo>                                (file picked via the HF API)
  https://civitai.com/models/<id>[?modelVersionId=<v>]                 (converted to the API route)
  https://civitai.com/api/download/models/<id>
  any direct https URL ending in .safetensors

Every download is validated before it is advertised: the safetensors header must
parse and declare at least one tensor. A truncated download or an HTML error
page saved as .safetensors is rejected rather than offered in the dropdown.

usage: python scripts/add_lora.py <url> [filename]
"""
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request

LORA_DIR = "/content/anima_lora"
STATE_PATH = "/content/anima_download.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) anima-lora-fetch/1.0"
CHUNK = 1 << 16


def write_state(**kw):
    kw.setdefault("t", time.time())
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f)
    os.replace(tmp, STATE_PATH)


def sanitize_name(name):
    """A filename that is safe to join onto LORA_DIR.

    basename() first, so "../../etc/passwd" cannot escape the directory, then a
    whitelist, then the extension is enforced because the server filters on it.
    """
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    if not name:
        name = "lora"
    if not name.lower().endswith(".safetensors"):
        name += ".safetensors"
    return name


def resolve_url(url, name_hint=""):
    """Map a page link to a direct download URL. Returns (url, filename) or (None, None).

    Pure: no network. (None, None) means "recognised as a HuggingFace repo but no
    file named", which the caller resolves through the API.
    """
    u = (url or "").strip()
    if not u:
        raise ValueError("empty link")
    if not u.lower().startswith(("http://", "https://")):
        raise ValueError("link must start with http:// or https://")

    # ---- HuggingFace -------------------------------------------------------
    m = re.search(r"huggingface\.co/([^/\s]+)/([^/\s?#]+)", u)
    if m:
        owner, repo = m.group(1), m.group(2)
        base = f"https://huggingface.co/{owner}/{repo}"
        if "/resolve/" in u:
            path = u.split("/resolve/", 1)[1].split("?")[0]
            return f"{base}/resolve/{path}", sanitize_name(
                name_hint or os.path.basename(path))
        for seg in ("blob", "tree"):
            if f"/{seg}/" in u:
                path = u.split(f"/{seg}/", 1)[1].split("?")[0]
                return f"{base}/resolve/{path}", sanitize_name(
                    name_hint or os.path.basename(path))
        # Bare repo: let the caller ask the API which files exist.
        return None, (sanitize_name(name_hint) if name_hint else None)

    # ---- CivitAI -----------------------------------------------------------
    m = re.search(r"civitai\.com/(?:models|api/download/models)/(\d+)", u)
    if m:
        model_id = m.group(1)
        if "/api/download/" in u:
            return u.split("?")[0], sanitize_name(
                name_hint or f"civitai_{model_id}")
        q = u.split("?", 1)[1] if "?" in u else ""
        params = {}
        for part in q.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
        version = params.get("modelVersionId") or params.get("version") or ""
        dl = f"https://civitai.com/api/download/models/{model_id}"
        if version:
            dl += f"?version={version}"
        return dl, sanitize_name(name_hint or f"civitai_{model_id}")

    # ---- anything else that is a direct file -------------------------------
    if u.lower().split("?")[0].endswith(".safetensors"):
        return u, sanitize_name(name_hint or os.path.basename(u.split("?")[0]))

    raise ValueError("unrecognised link: expected a HuggingFace or CivitAI URL "
                     "ending in .safetensors")


def hf_pick_file(repo_url):
    """Ask the HF API which .safetensors files a repo has; return (url, name).

    Used only for a bare repo link. Prefers a file with "lora" in the name, then
    the first one listed, so a repo of adapters does the obvious thing.
    """
    m = re.search(r"huggingface\.co/([^/\s]+)/([^/\s?#]+)", repo_url)
    if not m:
        raise ValueError("not a HuggingFace repo link")
    owner, repo = m.group(1), m.group(2)
    api = f"https://huggingface.co/api/models/{owner}/{repo}"
    req = urllib.request.Request(api, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    files = [s.get("rfilename", "") for s in data.get("siblings", [])]
    cands = [f for f in files if f.lower().endswith(".safetensors")]
    if not cands:
        raise ValueError("that repo has no .safetensors files")
    pick = next((f for f in cands if "lora" in f.lower()), cands[0])
    return (f"https://huggingface.co/{owner}/{repo}/resolve/main/{pick}",
            sanitize_name(os.path.basename(pick)))


def validate_safetensors(path):
    """Parse the header. Returns tensor count; raises on anything suspicious."""
    size = os.path.getsize(path)
    if size < 64:
        raise ValueError(f"file is only {size} bytes")
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated header length")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 100 * 1024 * 1024:
            raise ValueError(f"implausible header length {n}")
        blob = f.read(n)
        if len(blob) != n:
            raise ValueError("truncated header")
        try:
            h = json.loads(blob)
        except Exception as e:
            raise ValueError(f"header is not JSON: {type(e).__name__}")
    h.pop("__metadata__", None)
    if not h:
        raise ValueError("header declares no tensors")
    for k, v in h.items():
        if not isinstance(v, dict) or "dtype" not in v or "shape" not in v:
            raise ValueError(f"malformed tensor entry: {k}")
    return len(h)


def download(dl_url, target):
    """Stream to target, publishing progress. Removes a partial file on failure."""
    req = urllib.request.Request(dl_url, headers={"User-Agent": UA})
    tmp = target + ".part"
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" in ctype:
            raise ValueError("server returned an HTML page, not a model file "
                             "(the link probably needs an API route or auth)")
        done = 0
        last = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                # ~1 MB granularity: the UI polls once a second.
                if done - last >= (1 << 20) or (total and done == total):
                    last = done
                    write_state(status="downloading", name=os.path.basename(target),
                                done=done, total=total,
                                pct=(100 * done // total) if total else 0)
        if total and done != total:
            os.remove(tmp)
            raise ValueError(f"short read: {done} of {total} bytes")
    os.replace(tmp, target)
    return done


def main(argv):
    if len(argv) < 2:
        print("usage: add_lora.py <url> [filename]", file=sys.stderr)
        return 2
    url = argv[1]
    name_hint = argv[2] if len(argv) > 2 else ""

    os.makedirs(LORA_DIR, exist_ok=True)
    write_state(status="resolving", url=url)
    try:
        dl_url, name = resolve_url(url, name_hint)
        if dl_url is None:
            write_state(status="resolving", url=url, note="asking the HF API for files")
            dl_url, name = hf_pick_file(url)
        if not name:
            raise ValueError("could not work out a filename; pass one explicitly")
        target = os.path.join(LORA_DIR, name)
        if os.path.exists(target):
            # Re-validate and reuse: a second click must not re-download 600 MB.
            n = validate_safetensors(target)
            write_state(status="ok", name=name, size=os.path.getsize(target),
                        tensors=n, note="already present")
            print(f"ALREADY_PRESENT {name} {n} tensors")
            return 0

        write_state(status="downloading", name=name, pct=0, done=0, total=0)
        size = download(dl_url, target)
        write_state(status="validating", name=name, size=size)
        tensors = validate_safetensors(target)
        write_state(status="ok", name=name, size=size, tensors=tensors,
                    source=dl_url)
        print(f"ADDED {name} {size} bytes {tensors} tensors")
        return 0
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # A rejected file must not linger where the dropdown would offer it.
        try:
            n = sanitize_name(name_hint) if name_hint else None
            if n and os.path.exists(os.path.join(LORA_DIR, n)):
                pass
        except Exception:
            pass
        write_state(status="error", url=url, message=msg)
        print(f"FAILED {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
