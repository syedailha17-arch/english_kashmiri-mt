import argparse
import pandas as pd
import pandas.api.types
import sacrebleu
from KashmiriNormalizer import KashmiriNormalizer

_normalizer = KashmiriNormalizer()

def normalize_text(value):
    text = "" if pd.isna(value) else str(value)
    return _normalizer.normalize(text)

def evaluate(predictions_path, references_path, id_column="ID"):
    solution = pd.read_csv(references_path)
    submission = pd.read_csv(predictions_path)

    del solution[id_column]
    del submission[id_column]

    if len(submission) != len(solution):
        raise ValueError(f"Mismatch: {len(submission)} predictions vs {len(solution)} references.")

    references = [normalize_text(v) for v in solution["kashmiri_text"]]
    hypotheses = [normalize_text(v) for v in submission["kashmiri_text"]]

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
    geo_mean = (bleu * chrf) ** 0.5 if bleu > 0 and chrf > 0 else 0.0

    print(f"Predictions: {predictions_path}")
    print(f"References: {references_path}")
    print(f"Sentences scored: {len(submission)}")
    print("-" * 40)
    print(f"BLEU score: {bleu:.2f}")
    print(f"chrF++ score: {chrf:.2f}")
    print(f"Geometric Mean: {geo_mean:.2f}")

    return bleu, chrf, geo_mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translations with official KATHE scoring")
    parser.add_argument("--predictions", required=True, help="Path to submission CSV (ID, kashmiri_text)")
    parser.add_argument("--references", required=True, help="Path to reference CSV (ID, kashmiri_text)")
    args = parser.parse_args()
    evaluate(args.predictions, args.references)