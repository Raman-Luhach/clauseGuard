from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from clauseguard.data import download as dl
from clauseguard.errors import InputError
from clauseguard.ingestion.loader import load_pdf, load_text
from clauseguard.ingestion.pdf import extract_pdf
from tests.pdfgen import make_pdf

PARAGRAPH = (
    "These terms govern your use of the service. We may suspend or terminate your account at "
    "any time, for any reason, without prior notice. Any dispute will be resolved by binding "
    "arbitration and you waive any right to a class action. We may change these terms at any "
    "time and your continued use means that you accept the new terms."
)


def _wrapped(text: str, width: int = 80) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return [*lines, line]


PAGES = [
    PARAGRAPH,
    "These rules explain how we handle content. We may remove any content you post if we "
    "believe it breaks these terms, and we are not liable for any loss caused by the removal. "
    "You keep ownership of your content but grant us a worldwide licence to use it.",
    "These terms are governed by the laws of the State of California. Any claim must be "
    "brought in the courts of San Francisco County, and you consent to their jurisdiction. "
    "If any part of these terms is invalid, the rest remains in force.",
]


def _three_page_pdf() -> bytes:
    pages = []
    for n, body in enumerate(PAGES, start=1):
        pages.append(["ACME TERMS OF SERVICE", *_wrapped(body), f"Page {n} of 3"])
    return make_pdf(pages)


def test_extract_removes_running_headers_and_page_numbers():
    pdf = extract_pdf(_three_page_pdf())
    assert pdf.n_pages == 3
    assert "ACME TERMS OF SERVICE" not in pdf.text
    assert "Page 1 of 3" not in pdf.text
    for body in PAGES:
        assert body.split()[-1] in pdf.text


def test_loaded_pdf_unwraps_lines_and_knows_pages():
    doc = load_pdf(_three_page_pdf())
    assert doc.source == "pdf" and doc.n_pages == 3
    assert "for any reason, without prior notice." in doc.text
    positions = [doc.text.index(" ".join(body.split()[:6])) for body in PAGES]
    assert [doc.page_of(p) for p in positions] == [1, 2, 3]
    assert doc.page_of(len(doc.text) - 1) == 3


def test_scanned_pdf_is_rejected():
    with pytest.raises(InputError) as err:
        load_pdf(make_pdf([[], [], []]))
    assert err.value.code == "scanned_pdf"


def test_partly_empty_pdf_gives_a_warning():
    doc = load_pdf(make_pdf([_wrapped(PARAGRAPH * 2), [], _wrapped(PARAGRAPH * 2)]))
    assert any("1 of 3 pages had no selectable text" in w for w in doc.warnings)


def test_broken_pdf_is_rejected():
    with pytest.raises(InputError) as err:
        load_pdf(b"%PDF-1.4 this is not really a pdf")
    assert err.value.code == "unreadable_pdf"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", "empty_input"),
        ("We may terminate your account.", "too_short"),
        (
            "Nous pouvons résilier votre compte à tout moment, pour quelque raison que ce soit, "
            "sans préavis. Tout litige sera tranché par un arbitrage contraignant et vous "
            "renoncez à toute action collective. Nous pouvons modifier ces conditions à tout "
            "moment et votre utilisation continue signifie que vous acceptez les nouvelles "
            "conditions. Ces conditions sont régies par le droit français.",
            "not_english",
        ),
    ],
)
def test_invalid_text_is_rejected(text, code):
    with pytest.raises(InputError) as err:
        load_text(text)
    assert err.value.code == code


def test_valid_text_loads():
    doc = load_text(PARAGRAPH)
    assert doc.source == "text" and doc.page_of(0) is None and doc.n_words > 50


# ---------------------------------------------------------------- downloads


def _serve(tmp_path, monkeypatch, name: str, payload: bytes, *, extract: bool = False):
    src = tmp_path / "upstream" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(payload)
    manifest = {
        "demo": {
            "description": "demo",
            "license": "none",
            "files": [
                {
                    "url": src.as_uri(),
                    "filename": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "extract": extract,
                }
            ],
        }
    }
    monkeypatch.setattr(dl, "load_manifest", lambda: manifest)
    return manifest


def test_download_verifies_and_is_idempotent(home, tmp_path, monkeypatch):
    _serve(tmp_path, monkeypatch, "data.txt", b"hello")
    target = dl.download("demo")
    assert (target / "data.txt").read_bytes() == b"hello"
    stamp = (target / ".complete").stat().st_mtime_ns
    dl.download("demo")  # second call is a no-op
    assert (target / ".complete").stat().st_mtime_ns == stamp


def test_download_rejects_wrong_checksum(home, tmp_path, monkeypatch):
    manifest = _serve(tmp_path, monkeypatch, "data.txt", b"hello")
    manifest["demo"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dl.download("demo")
    assert not (home / "data" / "raw" / "demo" / "data.txt").exists()


def test_zip_extraction_skips_macos_junk_and_blocks_path_escape(home, tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("corpus/a.txt", "a")
        zf.writestr("__MACOSX/corpus/._a.txt", "junk")
    _serve(tmp_path, monkeypatch, "c.zip", buf.getvalue(), extract=True)
    target = dl.download("demo")
    assert (target / "corpus" / "a.txt").read_text() == "a"
    assert not (target / "__MACOSX").exists()

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    _serve(tmp_path, monkeypatch, "evil.zip", evil.getvalue(), extract=True)
    with pytest.raises(RuntimeError, match="unsafe path"):
        dl.download("demo", force=True)


def test_resolve_targets_expands_groups():
    assert dl.resolve_targets(["tos"]) == ["unfair_tos", "claudette"]
    assert dl.resolve_targets(["all", "cuad"]) == ["unfair_tos", "claudette", "opp115", "cuad"]
    with pytest.raises(ValueError, match="unknown dataset"):
        dl.resolve_targets(["nope"])
