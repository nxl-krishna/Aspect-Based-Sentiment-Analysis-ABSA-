"""Conversion between character-level aspect spans and token-level BIO labels."""

BIO_LABELS = ["O", "B-ASP", "I-ASP"]
O, B, I = 0, 1, 2
IGNORE = -100


def _trim(text, start, end):
    # SentencePiece/BPE offsets can include the leading space of a word
    while start < end and text[start].isspace():
        start += 1
    return start, end


def spans_to_token_labels(text, offsets, spans):
    """Label each token B/I/O given its (start, end) character offsets.

    Special tokens (offset (0, 0)) get IGNORE so they don't count in the loss.
    """
    labels = []
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE)
            continue
        start, end = _trim(text, start, end)
        label = O
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                label = B if start <= span_start else I
                break
        labels.append(label)

    # A span whose first character falls inside a token would otherwise start with I
    for i, label in enumerate(labels):
        if label == I:
            prev = next((l for l in reversed(labels[:i]) if l != IGNORE), O)
            if prev == O:
                labels[i] = B
    return labels


def token_labels_to_spans(text, offsets, labels):
    """Decode predicted BIO labels into (start, end) character spans."""
    spans = []
    current = None
    for (start, end), label in zip(offsets, labels):
        if start == end:
            continue
        start, end = _trim(text, start, end)
        if label == B or (label == I and current is None):
            if current:
                spans.append(tuple(current))
            current = [start, end]
        elif label == I:
            current[1] = end
        else:
            if current:
                spans.append(tuple(current))
            current = None
    if current:
        spans.append(tuple(current))
    return spans


def token_label_spans(labels):
    """Token-index spans from a BIO label sequence (IGNORE positions skipped)."""
    spans, start, prev_idx = [], None, None
    for idx, label in enumerate(labels):
        if label == IGNORE:
            continue
        if label == B or (label == I and start is None):
            if start is not None:
                spans.append((start, prev_idx))
            start = idx
        elif label == O and start is not None:
            spans.append((start, prev_idx))
            start = None
        prev_idx = idx
    if start is not None:
        spans.append((start, prev_idx))
    return spans
