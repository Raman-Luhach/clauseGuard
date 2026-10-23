from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from clauseguard import pipeline
from clauseguard.data.units import save_units

# A few sentences per ToS label, plus neutral sentences, to build a tiny learnable dataset.
SYNTHETIC = {
    "tos:limitation_of_liability": [
        "we are not liable for any damages arising from your use of the service",
        "our total liability shall not exceed the amount you paid us",
    ],
    "tos:unilateral_termination": [
        "we may terminate your account at any time without notice",
        "we can suspend your access for any reason at our sole discretion",
    ],
    "tos:unilateral_change": [
        "we may modify these terms at any time and your continued use means acceptance",
        "we reserve the right to change these terms whenever we like",
    ],
    "tos:content_removal": [
        "we may remove any content you post without telling you",
        "we can delete your uploads that we find objectionable",
    ],
    "tos:contract_by_using": [
        "by using the service you agree to be bound by these terms",
        "accessing the website constitutes acceptance of this agreement",
    ],
    "tos:choice_of_law": [
        "these terms are governed by the laws of the state of california",
        "this agreement shall be governed by english law",
    ],
    "tos:jurisdiction": [
        "any dispute shall be brought exclusively in the courts of new york",
        "you submit to the exclusive jurisdiction of the courts of dublin",
    ],
    "tos:arbitration": [
        "any dispute will be resolved by binding individual arbitration",
        "you agree to arbitrate all claims and waive any class action",
    ],
}
NEUTRAL = [
    "you can contact our support team by email",
    "the service lets you share photos with your friends",
    "our office is open from monday to friday",
    "please read the frequently asked questions page",
    "you can change your profile picture in the settings menu",
    "we hope you enjoy using our application",
]
FILLER = ["also", "please note that", "in addition", "importantly", "for clarity", "generally"]


def _sentence(rng: random.Random, base: str) -> str:
    return f"{rng.choice(FILLER)} {base} ." if rng.random() < 0.5 else f"{base} ."


PRIVACY = {
    ("privacy:third_party_sharing", "privacy:sharing_ads_marketing"): [
        "We share your browsing data with advertising networks to show you targeted ads.",
        "Advertising partners receive your device identifiers for marketing campaigns.",
    ],
    ("privacy:third_party_sharing", "privacy:negated_practice"): [
        "We do not sell or rent your personal information to anyone.",
        "We never share your contact details with outside companies.",
    ],
    ("privacy:first_party_collection",): [
        "We collect your name, email address and phone number when you register.",
        "When you use the app we record your location and device model.",
    ],
    ("privacy:user_choice",): [
        "You can opt out of marketing emails at any time by clicking unsubscribe.",
        "You may choose not to receive promotional messages in your preferences.",
    ],
    ("privacy:user_access",): [
        "You can access, update or delete your account information in the settings.",
        "Request a copy of your data or ask us to erase it through the account page.",
    ],
    ("privacy:data_security",): [
        "We use encryption and secure servers to protect your information.",
        "Your data is protected with firewalls and strict access controls.",
    ],
    ("privacy:data_retention",): [
        "We keep your data for as long as your account remains active.",
        "Records are stored for seven years and then permanently deleted.",
    ],
    ("privacy:policy_change", "privacy:change_without_personal_notice"): [
        "We may update this policy and will post the changes on this page.",
        "Changes to this privacy policy are effective when published on our website.",
    ],
    ("privacy:international_audiences",): [
        "Our services are not directed to children under the age of thirteen.",
        "Residents of the European Union have additional rights described below.",
    ],
    ("privacy:other",): [
        "If you have questions please contact our privacy team at the address below.",
        "This policy explains how our company handles information about visitors.",
    ],
}
CONTRACT = {
    "contract:governing_law": [
        "This Agreement shall be governed by the laws of the State of Delaware.",
        "This Agreement is governed by and construed under the laws of New York.",
    ],
    "contract:non_compete": [
        "The Distributor shall not sell competing products in the Territory during the Term.",
        "Supplier agrees not to compete with the Company for two years after termination.",
    ],
    "contract:anti_assignment": [
        "Neither party may assign this Agreement without the prior written consent of the other.",
        "This Agreement may not be assigned or transferred without written consent.",
    ],
    "contract:audit_rights": [
        "The Company may audit the books and records of the Distributor once a year.",
        "Licensor shall have the right to inspect and audit Licensee records upon notice.",
    ],
}
CONTRACT_NEUTRAL = [
    "The parties have executed this Agreement as of the Effective Date.",
    "Capitalised terms have the meanings given to them in Section 1.",
    "The Distributor shall deliver monthly sales reports by email to the Company.",
    "Notices shall be sent to the addresses listed on the signature page.",
]


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["char_start"] = df["char_start"].astype("Int64")
    df["char_end"] = df["char_end"].astype("Int64")
    return df


def _row(doc: str, i: int, unit_type: str, text: str, labels: list[str], split: str) -> dict:
    return {
        "doc_id": doc,
        "unit_id": f"{doc}:{i}",
        "unit_type": unit_type,
        "text": text,
        "char_start": None,
        "char_end": None,
        "labels": labels,
        "split": split,
        "exclude_from_training": False,
        "meta": json.dumps({}),
    }


SPLITS = (("train", 6), ("validation", 3), ("test", 3))


def synthetic_tos_units(seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for split, n_docs in SPLITS:
        for d in range(n_docs):
            doc = f"{split}-doc{d}"
            sentences = [
                (_sentence(rng, s), [label]) for label, ss in SYNTHETIC.items() for s in ss
            ]
            sentences += [(_sentence(rng, s), []) for s in NEUTRAL * 2]
            rng.shuffle(sentences)
            rows += [
                _row(doc, i, "sentence", t, labs, split) for i, (t, labs) in enumerate(sentences)
            ]
    return _frame(rows)


def synthetic_privacy_units(seed: int = 0) -> pd.DataFrame:
    """Paragraph units: each paragraph is one template sentence plus a neutral sentence."""
    rng = random.Random(seed)
    rows = []
    for split, n_docs in SPLITS:
        for d in range(n_docs):
            doc = f"{split}-policy{d}"
            paragraphs = [
                (f"{s} {rng.choice(NEUTRAL).capitalize()}.", list(labels))
                for labels, ss in PRIVACY.items()
                for s in ss
            ]
            rng.shuffle(paragraphs)
            rows += [
                _row(doc, i, "segment", t, labs, split) for i, (t, labs) in enumerate(paragraphs)
            ]
    return _frame(rows)


def synthetic_contract_units(seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for split, n_docs in SPLITS:
        for d in range(n_docs):
            doc = f"{split}-contract{d}"
            sentences = [(s, [label]) for label, ss in CONTRACT.items() for s in ss]
            sentences += [(s, []) for s in CONTRACT_NEUTRAL * 2]
            rng.shuffle(sentences)
            rows += [
                _row(doc, i, "sentence", t, labs, split) for i, (t, labs) in enumerate(sentences)
            ]
    return _frame(rows)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty project home: data, artifacts and logs go to a temporary directory."""
    monkeypatch.setenv("CLAUSEGUARD_HOME", str(tmp_path))
    monkeypatch.delenv("CLAUSEGUARD_LLM", raising=False)
    pipeline.get_analyzer.cache_clear()
    pipeline.get_explainer.cache_clear()
    pipeline.clear_cache()
    yield tmp_path
    pipeline.get_analyzer.cache_clear()
    pipeline.get_explainer.cache_clear()


@pytest.fixture
def synthetic_tos(home):
    units = synthetic_tos_units()
    save_units(units, "tos")
    return units


@pytest.fixture
def trained_tos(synthetic_tos):
    """A TF-IDF ToS model trained on the synthetic data; returns the run directory."""
    from clauseguard.models.train import train_run

    return train_run("tos", "tfidf", log=lambda *_: None)


@pytest.fixture
def trained_privacy(home):
    from clauseguard.models.train import train_run

    save_units(synthetic_privacy_units(), "privacy")
    return train_run("privacy", "tfidf", log=lambda *_: None)


@pytest.fixture
def trained_contract(home):
    from clauseguard.models.train import train_run

    save_units(synthetic_contract_units(), "contract")
    return train_run("contract", "tfidf", log=lambda *_: None)
