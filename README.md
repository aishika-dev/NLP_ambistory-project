# NLP_ambistory-project
## Dataset: SemEval 2026 Task 5 — AmbiStory

This project is part of the Introduction to Natural Language Processing course.
The task is to predict the human-perceived plausibility score of a candidate word sense in an ambiguous short story.

Each sample contains a short narrative with:

a three-sentence precontext,
an ambiguous sentence containing a target homonym,
an optional ending,
a candidate meaning,
an example sentence,
and a human average plausibility score from 1 to 5.

The system predicts a score between 1 and 5 for each input sample.

## Approach

This project implements a simple supervised regression baseline:

TF-IDF + Ridge Regression

The text fields are combined into one input string:

precontext + sentence + ending + judged_meaning + example_sentence

Then:

TF-IDF converts the text into numerical features.
Ridge Regression learns to predict the average human plausibility score.
The predicted score is clipped to the range 1–5 and rounded to an integer.
Predictions are saved in JSONL format.

This approach is lightweight, runs locally, and does not require external APIs.

## Project Structure

NLP_ambistory-project/
│
├── predict.py          # Main prediction script
├── evaluate.py         # Script for local evaluation on dev data
├── requirements.txt    # Required Python dependencies
├── README.md           # Project documentation
├── train.json          # Training dataset
├── dev.json            # Development dataset
└── output.jsonl        # Generated prediction output

## Setup Instructions

Clone the repository:

git clone https://github.com/aishika-dev/NLP_ambistory-project.git
cd NLP_ambistory-project

Install the required dependencies:

pip install -r requirements.txt

The main dependency used in this project is:

scikit-learn
Running the Prediction Script

The prediction script follows the required coding standard:

python predict.py <input_json> <output_jsonl>

Example:

python predict.py dev.json output.jsonl

This command reads the input JSON file, predicts plausibility scores for all samples, and writes the results into output.jsonl.

Output Format

The output file is in JSONL format. Each line contains one prediction:

{"id": "0", "prediction": 3}
{"id": "1", "prediction": 4}
{"id": "2", "prediction": 2}

Each output line contains:

id          - sample id from the input file
prediction  - predicted plausibility score between 1 and 5
Local Evaluation

To evaluate the predictions on the development set, first run:

python predict.py dev.json output.jsonl

Then run:

python evaluate.py

The evaluation script calculates:

Spearman Correlation
Accuracy Within Standard Deviation

These metrics compare the model predictions with the human average plausibility scores in the development data.

## Dataset

The project uses JSON files containing story samples. Each sample includes:

precontext
sentence
ending
homonym
judged_meaning
example_sentence
average
stdev

The average field is used as the training target because it represents the average human plausibility judgment.
