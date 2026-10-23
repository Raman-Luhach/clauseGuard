"""Download, verify and extract the datasets listed in ``manifest.yaml``."""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

import certifi
import yaml
from tqdm import tqdm

from clauseguard.paths import raw_dir

MANIFEST = Path(__file__).with_name("manifest.yaml")
GROUPS = {
    "tos": ["unfair_tos", "claudette"],
    "privacy": ["opp115"],
    "contract": ["cuad"],
}
GROUPS["all"] = [name for names in GROUPS.values() for name in names]
_USER_AGENT = "clauseguard/0.1 (academic research project)"
_RETRIES = 4
# certifi's CA bundle: some Python builds (e.g. python.org's on macOS) ship without one.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def load_manifest() -> dict[str, dict]:
    with open(MANIFEST, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_targets(targets: list[str]) -> list[str]:
    manifest = load_manifest()
    names: list[str] = []
    for target in targets:
        if target in GROUPS:
            names.extend(GROUPS[target])
        elif target in manifest:
            names.append(target)
        else:
            choices = sorted(GROUPS) + sorted(manifest)
            raise ValueError(f"unknown dataset {target!r}; choose from {', '.join(choices)}")
    return list(dict.fromkeys(names))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, expected_sha256: str) -> None:
    part = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with (
                urllib.request.urlopen(request, timeout=60, context=_SSL_CONTEXT) as response,
                open(part, "wb") as out,
            ):
                total = int(response.headers.get("Content-Length") or 0) or None
                with tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in iter(lambda: response.read(1 << 20), b""):
                        out.write(chunk)
                        bar.update(len(chunk))
            break
        except OSError as exc:  # URLError, timeouts, SSL errors, connection resets
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < _RETRIES:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(
            f"failed to download {url} after {_RETRIES} attempts: {last_error}"
        ) from last_error

    actual = sha256_of(part)
    if actual != expected_sha256:
        part.unlink()
        raise RuntimeError(
            f"checksum mismatch for {url}\n  expected {expected_sha256}\n  got      {actual}\n"
            "The upstream file has changed; inspect it before updating manifest.yaml."
        )
    part.replace(dest)


def _extract_zip(archive: Path, target: Path, skip: list[str] | None = None) -> None:
    """Extract, skipping macOS metadata and unused folders (``skip`` path prefixes), and
    refusing paths that escape ``target``."""
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if any(member.filename.startswith(prefix) for prefix in skip or ()):
                continue
            parts = Path(member.filename).parts
            if (
                not parts
                or parts[0] == "__MACOSX"
                or parts[-1].startswith("._")
                or parts[-1] == ".DS_Store"
            ):
                continue
            dest = (target / member.filename).resolve()
            if not dest.is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive.name}: {member.filename}")
            if member.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)


def download(name: str, *, force: bool = False) -> Path:
    spec = load_manifest()[name]
    target = raw_dir(name)
    marker = target / ".complete"
    stamp = json.dumps({f["filename"]: f["sha256"] for f in spec["files"]}, sort_keys=True)
    if not force and marker.exists() and marker.read_text() == stamp:
        return target

    target.mkdir(parents=True, exist_ok=True)
    for file in spec["files"]:
        dest = target / file["filename"]
        if force or not dest.exists() or sha256_of(dest) != file["sha256"]:
            _fetch(file["url"], dest, file["sha256"])
        if file.get("extract"):
            _extract_zip(dest, target, file.get("skip"))
    marker.write_text(stamp)
    return target
