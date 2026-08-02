# Load BPCC dataset
from datasets import load_dataset
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from sacrebleu import BLEU, CHRF
import math

print("Loading BPCC Kashmiri dataset...")
dataset = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest", split="kas_Arab")
print(f"Dataset size: {len(dataset)}\n")

# Load NLLB-200 baseline (no custom code needed)
print("Loading NLLB-200 baseline model...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
print(f"Model loaded\n")

# Generate translations on first 100 examples
print("Generating translations on first 100 examples...")
predictions = []
references = []

for i in range(min(100, len(dataset))):
    english = dataset[i]['src']
    kashmiri_ref = dataset[i]['tgt']
    
    # Tokenize and generate
    inputs = tokenizer(english, return_tensors="pt")
    generated_ids = model.generate(
        inputs["input_ids"], 
        max_length=128,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kas_Arab")
    )
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    predictions.append(prediction)
    references.append(kashmiri_ref)
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1} examples...")

print(f"\nGenerated {len(predictions)} predictions\n")

# Score with BLEU and chrF++
print("Computing BLEU and chrF++...")
bleu = BLEU()
chrf = CHRF()

bleu_score = bleu.corpus_score(predictions, [references])
chrf_score = chrf.corpus_score(predictions, [references])

# Geometric mean
geo_mean = math.sqrt(bleu_score.score * chrf_score.score)

print(f"\n{'='*50}")
print(f"BASELINE SCORES (NLLB-200 on 100 examples)")
print(f"{'='*50}")
print(f"BLEU:            {bleu_score.score:.2f}")
print(f"chrF++:          {chrf_score.score:.2f}")
print(f"Geometric mean:  {geo_mean:.2f}")
print(f"{'='*50}\n")

# Show a few examples
print("--- Sample Predictions ---")
for i in range(3):
    print(f"\n{i+1}. English: {dataset[i]['src'][:60]}...")
    print(f"   Reference: {references[i][:60]}...")
    print(f"   Prediction: {predictions[i][:60]}...")