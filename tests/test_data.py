from src.data.bio import B, I, IGNORE, O, spans_to_token_labels, token_label_spans, token_labels_to_spans
from src.data.semeval import parse_sentences, sentences_to_pairs

XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="1">
    <text>The battery life is great but the screen is dim.</text>
    <aspectTerms>
      <aspectTerm term="battery life" polarity="positive" from="4" to="16"/>
      <aspectTerm term="screen" polarity="negative" from="34" to="40"/>
    </aspectTerms>
  </sentence>
  <sentence id="2">
    <text>I bought it yesterday.</text>
  </sentence>
  <sentence id="3">
    <text>Service was fine and slow.</text>
    <aspectTerms>
      <aspectTerm term="Service" polarity="conflict" from="0" to="7"/>
    </aspectTerms>
  </sentence>
</sentences>
"""

TEXT = "The battery life is great"
# Offsets as a SentencePiece tokenizer reports them (leading spaces included), plus [CLS]/[SEP]
OFFSETS = [(0, 0), (0, 3), (3, 11), (11, 16), (16, 19), (19, 25), (0, 0)]


def test_parse_keeps_sentences_without_aspects(tmp_path):
    path = tmp_path / "sample.xml"
    path.write_text(XML, encoding="utf-8")
    sentences = parse_sentences(str(path))
    assert len(sentences) == 3
    assert sentences[1]["aspects"] == []
    assert sentences[0]["aspects"][0] == {"term": "battery life", "from": 4, "to": 16, "polarity": "positive"}


def test_pairs_drop_conflict_polarity(tmp_path):
    path = tmp_path / "sample.xml"
    path.write_text(XML, encoding="utf-8")
    pairs = sentences_to_pairs(parse_sentences(str(path)), domain="Laptops")
    assert pairs["aspect"].tolist() == ["battery life", "screen"]
    assert set(pairs["domain"]) == {"Laptops"}


def test_spans_to_labels_handles_leading_space_offsets():
    labels = spans_to_token_labels(TEXT, OFFSETS, [(4, 16)])
    assert labels == [IGNORE, O, B, I, O, O, IGNORE]


def test_labels_round_trip_to_character_spans():
    labels = spans_to_token_labels(TEXT, OFFSETS, [(4, 16)])
    assert token_labels_to_spans(TEXT, OFFSETS, labels) == [(4, 16)]


def test_span_starting_mid_token_still_gets_b_tag():
    labels = spans_to_token_labels(TEXT, OFFSETS, [(6, 16)])
    assert labels[2] == B


def test_orphan_inside_tag_starts_new_span():
    assert token_labels_to_spans(TEXT, OFFSETS, [IGNORE, O, I, I, O, O, IGNORE]) == [(4, 16)]


def test_token_label_spans_skips_ignored_positions():
    assert token_label_spans([IGNORE, B, I, O, B, IGNORE]) == [(1, 2), (4, 4)]
