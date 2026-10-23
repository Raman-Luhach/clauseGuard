from __future__ import annotations

import random

import pytest

from clauseguard.ingestion.clean import clean_text
from clauseguard.ingestion.normalize import unfair_tos_style
from clauseguard.ingestion.segment import looks_like_heading, split_sentences
from clauseguard.textmap import TextWithMap, strip_html

# ---------------------------------------------------------------- TextWithMap


def test_identity_and_spans():
    tm = TextWithMap.identity("hello world")
    assert tm.to_source_span(6, 11) == (6, 11)
    assert tm.from_source_span(6, 11) == (6, 11)


def test_replace_keeps_origins_for_deletions_and_insertions():
    source = "a  b  c"
    tm = TextWithMap.identity(source).replace(r"[  ]+", " ")
    assert tm.text == "a b c"
    # every surviving character maps back onto the same character in the source
    for i, ch in enumerate(tm.text):
        if ch != " ":
            assert source[tm.origin[i]] == ch
    inserted = TextWithMap.identity("1.Amendment").replace(r"(?<=\d\.)(?=[A-Z])", " ")
    assert inserted.text == "1. Amendment"
    start, end = inserted.from_source_span(2, 11)  # "Amendment" in the source
    assert inserted.text[start:end] == "Amendment"


def test_origins_are_monotonic_after_cleaning():
    raw = "TERMS\n\nWe may   terminate your\naccount at any time.\n\n\n(a) first; and\n(b) second."
    tm = clean_text(raw)
    assert list(tm.origin) == sorted(tm.origin)


def test_strip_html_maps_spans_back_to_raw_html():
    html = "<p>We <b>share</b> your data with <i>partners</i> &amp; advertisers.</p>"
    tm = strip_html(html)
    assert tm.text.strip() == "We share your data with partners & advertisers."
    start = html.index("share")
    s, e = tm.from_source_span(start, start + len("share"))
    assert tm.text[s:e] == "share"
    # a span covering tags shrinks to the visible text inside it
    start = html.index("<i>")
    s, e = tm.from_source_span(start, html.index("</i>") + 4)
    assert tm.text[s:e] == "partners"


# ---------------------------------------------------------------- cleaning


def test_clean_unwraps_hard_wrapped_lines_but_keeps_structure():
    raw = (
        "TERMS OF SERVICE\n\n"
        "1. Termination. We may suspend or terminate your account at any time, for\n"
        "any reason, without prior notice.\n"
        "(a) the Licensee shall pay the fees; and\n"
        "(b) the Licensor shall deliver the third-\nparty Software."
    )
    text = clean_text(raw).text
    assert "for any reason, without prior notice." in text
    assert "\n(a) the Licensee" in text and "\n(b) the Licensor" in text
    assert "third-party Software" in text
    assert text.startswith("TERMS OF SERVICE\n\n1. Termination.")


def test_clean_normalises_characters():
    text = clean_text("“We” won’t — ever sell​ data…").text
    assert text == '"We" won\'t - ever sell data...'


def test_clean_separates_glued_list_markers():
    text = clean_text("a.As of the Closing date\nArticle 1.Amendment and Termination").text
    assert "a. As of" in text and "Article 1. Amendment" in text


def test_clean_offsets_point_at_the_same_characters():
    raw = "We may   change\nthese terms.\n\n\nYou agree."
    tm = clean_text(raw)
    rng = random.Random(0)
    for _ in range(50):
        i = rng.randrange(len(tm.text))
        if tm.text[i].isalnum():
            assert raw[tm.origin[i]] == tm.text[i]


# ---------------------------------------------------------------- segmentation

LEGAL = (
    "Acme Inc. may suspend or terminate your account at any time, for any reason. "
    "Subject to Section 4.2(b), you agree to binding arbitration. See e.g. the AAA rules. "
    "No. 5 of the schedule applies. This Agreement is governed by the laws of New York, "
    "U.S.A. without regard to conflict of laws."
)


def test_sentences_respect_legal_abbreviations():
    texts = [s.text for s in split_sentences(LEGAL)]
    assert texts == [
        "Acme Inc. may suspend or terminate your account at any time, for any reason.",
        "Subject to Section 4.2(b), you agree to binding arbitration.",
        "See e.g. the AAA rules.",
        "No. 5 of the schedule applies.",
        "This Agreement is governed by the laws of New York, U.S.A. without regard to "
        "conflict of laws.",
    ]


def test_sentence_offsets_index_into_text():
    text = clean_text("HEADER\n\n" + LEGAL + "\n(a) first item; and\n(b) second item.").text
    for s in split_sentences(text):
        assert text[s.start : s.end] == s.text
        assert s.text == s.text.strip()


def test_lines_are_never_merged_and_fragments_are():
    text = "Termination\n7. Arbitration. Disputes go to binding arbitration under the rules."
    texts = [s.text for s in split_sentences(text)]
    assert texts[0] == "Termination"
    assert texts[1].startswith("7. Arbitration.")


def test_long_units_are_split_at_semicolons():
    clause = "the supplier shall deliver the goods on time and in full"
    text = "; ".join([clause] * 60) + "."
    sentences = split_sentences(text, max_chars=500)
    assert len(sentences) > 1
    assert all(len(s.text) <= 500 for s in sentences)
    assert "".join(text[s.start : s.end] for s in sentences).replace(" ", "") == text.replace(
        " ", ""
    )


def test_all_lowercase_text_is_still_split():
    text = "we may terminate your account. you agree to arbitration. this is governed by law."
    assert len(split_sentences(text)) == 3


# ---------------------------------------------------------------- UNFAIR-ToS style


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            'We may terminate your account ("Account") at any time.',
            "we may terminate your account ( `` account '' ) at any time .",
        ),
        (
            "You can't sue; disputes over $75,000 go to arbitration.",
            "you ca n't sue ; disputes over $ 75,000 go to arbitration .",
        ),
        (
            "Acme's terms - see https://acme.com/terms: they apply.",
            "acme 's terms -- see https://acme.com/terms : they apply .",
        ),
    ],
)
def test_unfair_tos_style(raw, expected):
    assert unfair_tos_style(raw) == expected


def test_unfair_tos_style_is_idempotent():
    for raw in ['He said "no" (twice).', "We won't; ever.", "Fees: $5,000 - due now."]:
        once = unfair_tos_style(raw)
        assert unfair_tos_style(once) == once


@pytest.mark.parametrize(
    ("text", "heading"),
    [
        ("3. GitHub May Terminate", True),
        ("1. Subject to Change", True),
        ("LIMITATION OF LIABILITY", True),
        ("(iv)", True),
        ("We may terminate your account at any time.", False),
        ("(a) the Licensee shall pay the fees; and", False),
        ("Short version: Beta Previews may change at any time.", False),
        ("We Reserve The Right To Change These Terms At Any Time Without Any Notice To You", False),
    ],
)
def test_looks_like_heading(text, heading):
    assert looks_like_heading(text) is heading
