import json
from scipy.stats import spearmanr

with open("dev.json", "r", encoding="utf-8") as f:
    dev = json.load(f)

predictions = {}

with open("output.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        predictions[item["id"]] = item["prediction"]

gold = []
pred = []
within_std = 0

for sample_id, sample in dev.items():
    gold_score = float(sample["average"])
    pred_score = float(predictions[str(sample_id)])
    stdev = max(1.0, float(sample["stdev"]))

    gold.append(gold_score)
    pred.append(pred_score)

    if abs(pred_score - gold_score) <= stdev:
        within_std += 1

spearman, _ = spearmanr(gold, pred)
accuracy_within_std = within_std / len(dev)

print("Spearman Correlation:", spearman)
print("Accuracy Within Standard Deviation:", accuracy_within_std)