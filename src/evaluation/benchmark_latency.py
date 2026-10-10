"""Measure end-to-end pipeline latency (extract aspects + classify each) on CPU and GPU.

    python -m src.evaluation.benchmark_latency
"""
import os
import statistics
import time

import torch

from src.data.semeval import load_sentences
from src.inference.infer import ABSA_Inferencer, ABSAPipeline, AspectExtractor
from src.training.utils import REPORTS_DIR, save_json


def time_pipeline(pipeline, texts, single_n=100, batch_size=32):
    pipeline.analyze_batch(texts[:8])  # warm-up
    single = []
    for text in texts[:single_n]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        pipeline.analyze(text)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    pipeline.analyze_batch(texts, batch_size=batch_size)
    throughput = len(texts) / (time.perf_counter() - start)
    single.sort()
    return {
        "p50_ms": statistics.median(single),
        "p95_ms": single[int(0.95 * (len(single) - 1))],
        "batch_reviews_per_s": throughput,
        "n_batch": len(texts),
    }


def main():
    texts = [s["text"] for s in load_sentences("Restaurants", "Test")] + [s["text"] for s in load_sentences("Laptops", "Test")]
    pipeline = ABSAPipeline(sentiment=ABSA_Inferencer(), extractor=AspectExtractor())
    results = {}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for component in (pipeline.sentiment, pipeline.extractor):
            component.device = torch.device(device)
            component.model.to(device)
        name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        results[name] = time_pipeline(pipeline, texts if device == "cuda" else texts[:400], single_n=100 if device == "cuda" else 50)
        print(name, {k: round(v, 1) for k, v in results[name].items()})
    save_json(results, os.path.join(REPORTS_DIR, "latency.json"))


if __name__ == "__main__":
    main()
