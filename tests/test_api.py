import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api
from src.inference.infer import ABSAPipeline


class FakeSentiment:
    model_name = "fake-sentiment"

    def _label(self, aspect):
        return "Negative" if aspect.lower() in {"screen", "service"} else "Positive"

    def predict(self, text, aspect):
        return {"text": text, "aspect": aspect, "sentiment": self._label(aspect), "confidence": 99.0}

    def predict_pairs(self, texts, aspects, batch_size=32):
        return [self.predict(t, a) for t, a in zip(texts, aspects)]

    def batch_predict(self, df, batch_size=32):
        df = df[["text", "aspect"]].dropna()
        return pd.DataFrame(self.predict_pairs(df["text"].tolist(), df["aspect"].tolist()))


class FakeExtractor:
    model_name = "fake-extractor"
    VOCAB = ("battery", "screen", "service", "food")

    def extract(self, texts, batch_size=32):
        out = []
        for text in texts:
            lower = text.lower()
            out.append([{"aspect": text[i:i + len(w)], "from": i, "to": i + len(w)}
                        for w in self.VOCAB if (i := lower.find(w)) >= 0])
        return out


@pytest.fixture
def client(monkeypatch):
    sentiment = FakeSentiment()
    monkeypatch.setattr(api, "inferencer", sentiment)
    monkeypatch.setattr(api, "pipeline", ABSAPipeline(sentiment=sentiment, extractor=FakeExtractor()))
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json()["extractor_model"] == "fake-extractor"


def test_health_reports_unloaded_model(monkeypatch):
    monkeypatch.setattr(api, "inferencer", None)
    assert TestClient(api.app).get("/health").status_code == 503


def test_analyze_single_aspect(client):
    r = client.post("/analyze", json={"text": "The screen is dim.", "aspect": "screen"})
    assert r.status_code == 200
    assert r.json()["sentiment"] == "Negative"


def test_analyze_rejects_empty_input(client):
    assert client.post("/analyze", json={"text": "", "aspect": "screen"}).status_code == 422


def test_analyze_review_extracts_all_aspects(client):
    r = client.post("/analyze-review", json={"text": "Great battery but the screen is dim."})
    aspects = {a["aspect"]: a["sentiment"] for a in r.json()["aspects"]}
    assert aspects == {"battery": "Positive", "screen": "Negative"}


def test_batch_with_aspect_column(client):
    csv = "text,aspect\nGreat food,food\nRude service,service\n"
    r = client.post("/batch-analyze", files={"file": ("reviews.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert r.json()["summary"]["sentiment_breakdown"] == {"Positive": 1, "Negative": 1}


def test_batch_without_aspect_column_extracts_aspects(client):
    csv = "text\nGreat battery but the screen is dim.\nThe food was lovely.\n\n"
    r = client.post("/batch-analyze", files={"file": ("reviews.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert r.json()["summary"]["total_analyzed"] == 3


def test_batch_missing_text_column_is_client_error(client):
    r = client.post("/batch-analyze", files={"file": ("reviews.csv", "review\nhello\n", "text/csv")})
    assert r.status_code == 400


def test_batch_rejects_non_csv(client):
    r = client.post("/batch-analyze", files={"file": ("reviews.txt", "text\nhi\n", "text/plain")})
    assert r.status_code == 400
