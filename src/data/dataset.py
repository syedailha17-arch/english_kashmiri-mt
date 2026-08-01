import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class KashmiriTranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="ai4bharat/indictrans2-en-indic-1B", max_length=128):
        """
        PyTorch Dataset for Babar's ML Training Loop.
        Reads the clean CSVs and tokenizes them on the fly.
        """
        print(f"Loading data from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
        
        # Load the tokenizer (Defaulting to IndicTrans2 since it's SOTA for Indian languages)
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Grab the English source and Kashmiri target strings
        english_text = str(self.data.iloc[idx]['src'])
        kashmiri_text = str(self.data.iloc[idx]['tgt'])
        
        # Tokenize English (Source)
        source_encodings = self.tokenizer(
            english_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize Kashmiri (Target)
        target_encodings = self.tokenizer(
            kashmiri_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the tensors Babar needs for the PyTorch training loop
        return {
            'input_ids': source_encodings['input_ids'].flatten(),
            'attention_mask': source_encodings['attention_mask'].flatten(),
            'labels': target_encodings['input_ids'].flatten()
        }

# --- Quick Test to ensure it works for Babar ---
if __name__ == "__main__":
    print("Testing the PyTorch Dataset class...")
    # Create the dataset using the validation split
    val_dataset = KashmiriTranslationDataset(csv_file="data/processed/val.csv")
    
    # Create a DataLoader (how PyTorch batches data during training)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    # Fetch one batch to prove it works
    batch = next(iter(val_loader))
    print("\n--- Successful Batch Generation ---")
    print(f"Input IDs shape (English): {batch['input_ids'].shape}")
    print(f"Labels shape (Kashmiri):   {batch['labels'].shape}")
    print("Hand-off to Babar is ready!")
