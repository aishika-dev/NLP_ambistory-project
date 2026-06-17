import sys
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def make_text(sample):
    return " ".join([
        sample.get("precontext", ""),
        sample.get("sentence", ""),
        sample.get("ending", ""),
        sample.get("judged_meaning", ""),
        sample.get("example_sentence", "")
    ])


def load_train_data():
    with open("train.json", "r", encoding="utf-8") as f:
        return json.load(f)


def train_model(train_data):
    X = []
    y = []

    for _, sample in train_data.items():
        X.append(make_text(sample))
        y.append(float(sample["average"]))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            max_features=20000,
            min_df=1,
            sublinear_tf=True
        )),
        ("ridge", Ridge(alpha=0.1))
    ])

    model.fit(X, y)
    return model


def convert_prediction(score):
    score = max(1.0, min(5.0, float(score)))
    return int(round(score))


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_json> <output_jsonl>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_jsonl = sys.argv[2]

    train_data = load_train_data()
    model = train_model(train_data)

    with open(input_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for sample_id, sample in test_data.items():
            text = make_text(sample)
            raw_prediction = model.predict([text])[0]
            prediction = convert_prediction(raw_prediction)

            f_out.write(json.dumps({
                "id": str(sample_id),
                "prediction": prediction
            }) + "\n")


if __name__ == "__main__":
    main()