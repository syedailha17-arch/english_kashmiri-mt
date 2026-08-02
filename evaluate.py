"""
evaluate.py: scores translation preductions against reference translations using BLEU and chrF++ (via Sacrebleu)
Usage: -python evaluate.py
       -predictions preds.txt
       -references refs.txt
"""
import argparse
import sacrebleu

def load_lines(path):
    with open(path,"r",encoding="utf-8") as f:
        return[line.strip() for line in f if line.strip()]
def evaluate(predictions_path, references_path):
    predictions= load_lines(predictions_path)
    references=load_lines(references_path)
    if len(predictions)!=len(references):
        raise ValueError(
            f"Mismatch:{len(predictions)} predictions vs"
            f"{len(references)} references. They must be the same length"
            f"and line aligned(line 1 of predictions matches line 1 of references,etc.)"
        )

#Sacrebleu expectes references as a list of reference-lists
    refs_formatted=[references]
    bleu=sacrebleu.corpus_bleu(predictions, refs_formatted)
    chrf=sacrebleu.corpus_chrf(predictions, refs_formatted, word_order=2)

    print(f"Predictions:{predictions_path}")
    print(f"References:{references_path}")
    print(f"Sentences scored:{len(predictions)}")
    print("-"*40)
    print(f"BLEU score: {bleu.score:.2f}")
    print(f"chrF++ score: {chrf.score:.2f}")
    return bleu.score, chrf.score

if __name__=="__main__":
    parser= argparse.ArgumentParser(description="Evaluate translations with BLEU + chrF++")
    parser.add_argument("--predictions", required=True, help="Path to model output text file")
    parser.add_argument("--references", required=True, help="Path to reference (gold) translation text file")
    args= parser.parse_args()
    
    evaluate(args.predictions, args.references)
