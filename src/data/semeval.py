"""Loading utilities for the SemEval-2014 Task 4 (aspect term) datasets."""
import os
import xml.etree.ElementTree as ET

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

DOMAINS = ("Restaurants", "Laptops")
SPLITS = ("Train", "Test")
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def parse_sentences(path):
    """Parse a SemEval-2014 XML file into sentences with their aspect term spans.

    Every sentence is kept (including those without aspects) because the
    extraction model needs negative examples. Conflict polarities are kept here
    and filtered later by the sentiment loader.
    """
    root = ET.parse(path).getroot()
    sentences = []
    for sentence in root.iter("sentence"):
        aspects = []
        terms = sentence.find("aspectTerms")
        if terms is not None:
            for term in terms.iter("aspectTerm"):
                aspects.append({
                    "term": term.get("term"),
                    "from": int(term.get("from")),
                    "to": int(term.get("to")),
                    "polarity": term.get("polarity"),
                })
        sentences.append({"id": sentence.get("id"), "text": sentence.find("text").text, "aspects": aspects})
    return sentences


def load_sentences(domain, split, raw_dir=RAW_DIR):
    return parse_sentences(os.path.join(raw_dir, f"{domain}_{split}.xml"))


def sentences_to_pairs(sentences, domain=None):
    """Flatten sentences into one row per (sentence, aspect) with a 3-way polarity."""
    rows = []
    for s in sentences:
        for a in s["aspects"]:
            if a["polarity"] in LABEL2ID:
                rows.append({
                    "text": s["text"],
                    "aspect": a["term"],
                    "from": a["from"],
                    "to": a["to"],
                    "sentiment": a["polarity"],
                    "domain": domain,
                })
    return pd.DataFrame(rows, columns=["text", "aspect", "from", "to", "sentiment", "domain"])


def load_pairs(split, domains=DOMAINS, raw_dir=RAW_DIR):
    return pd.concat(
        [sentences_to_pairs(load_sentences(d, split, raw_dir), d) for d in domains],
        ignore_index=True,
    )
