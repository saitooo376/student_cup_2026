import json

def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )

# seve_submissionも後で追加