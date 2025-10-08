import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW 
from tqdm import tqdm
import pandas as pd
import os
import config
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

torch.cuda.empty_cache()

clarity_mapping ={
    '1.1 Explicit': 'Direct Reply',
    '1.2 Implicit': 'Indirect',
    '2.1 Dodging': "Indirect",
    '2.2 Deflection': "Indirect",
    '2.3 Partial/half-answer': "Indirect",
    '2.4 General': "Indirect",
    '2.6 Declining to answer': "Direct Non-Reply",
    '2.7 Claims ignorance': "Direct Non-Reply",
    '2.8 Clarification': "Direct Non-Reply",
}

class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=config.MAXLENGTH):
        self.max_length = max_length
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # First pass: check if truncation is needed
        inputs_check = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            truncation=False,  # Don't truncate yet, just check length
            add_special_tokens=True
        )
        
        original_length = len(inputs_check['input_ids'][0])
        is_truncated = original_length > self.max_length
        
        # Second pass: actually tokenize with truncation if needed
        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            truncation=True,  # Now truncate if necessary
            padding='max_length',
            max_length=self.max_length,
            add_special_tokens=True
        )
           
        label = torch.tensor(self.labels[idx])
        return inputs, label, is_truncated, original_length
    

dataset_path = "test_set.csv"
df = pd.read_csv(dataset_path)

def collate_fn(batch):
    inputs, labels, is_truncated, original_lengths = zip(*batch)
    return {
        'input_ids': torch.stack([x['input_ids'].squeeze() for x in inputs]),
        'attention_mask': torch.stack([x['attention_mask'].squeeze() for x in inputs]),
        'labels': torch.tensor(labels), 
        'is_truncated': is_truncated,
        'original_lengths': original_lengths
    }

if config.EXPERIMENTNAME == "evasion_based_clarity": 
    num_labels = 11
    mapping_labels = {'1.1 Explicit': 0, '1.2 Implicit': 1, '2.1 Dodging': 2, '2.2 Deflection': 3, '2.3 Partial/half-answer': 4, '2.4 General': 5, '2.5 Contradictory': 6, '2.6 Declining to answer': 7, '2.7 Claims ignorance': 8, '2.8 Clarification': 9, '2.9 Diffusion': 10}
elif config.EXPERIMENTNAME == "direct_clarity":
    num_labels = 3
    mapping_labels = {"Direct Reply": 0, "Indirect": 1, "Direct Non-Reply": 2}

from transformers import AlbertTokenizer, AlbertForSequenceClassification
tokenizer = AlbertTokenizer.from_pretrained(config.MODELNAME)
model = AlbertForSequenceClassification.from_pretrained(
    config.MODELNAME, 
    num_labels=num_labels
).to("cuda")

# Load your trained model for validation
model_path = config.OUTFILE  # Path where your trained model was saved
model = AlbertForSequenceClassification.from_pretrained(model_path).to("cuda")

labels = []

for _, row in df.iterrows():
    l = [row["Annotator1"], row["Annotator2"], row["Annotator3"]]
    labels.append(max(set(l), key=labels.count))
df["Label"] = labels

all_texts = [f"Question: {row['Interview Question']}\n\nAnswer: {row['Interview Answer']}\n\nSubanswer: {row['Question']}" for _, row in df.iterrows()]

if config.EXPERIMENTNAME == "evasion_based_clarity":
    all_labels = [mapping_labels[row["Label"]] for _, row in df.iterrows() if "other" not in row["Label"].lower()]
elif config.EXPERIMENTNAME == "direct_clarity":
    all_labels = [mapping_labels[clarity_mapping[row["Label"]]] for _, row in df.iterrows() if "other" not in row["Label"].lower()]

val_dataset = CustomDataset(all_texts, all_labels, max_length=config.MAXLENGTH)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model.eval()
inv_mapping_labels = {v:k for k, v in mapping_labels.items()}
results = []

true_labels, pred_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        inputs = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['labels'].to("cuda")
        is_truncated = batch['is_truncated']
        original_lengths = batch['original_lengths']
        
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        
        for true_label, logits, is_trunc, orig_len in zip(labels.cpu().numpy(), outputs.logits.cpu().numpy(), is_truncated, original_lengths):
            true_label_str = inv_mapping_labels[true_label]
            pred_label_str = inv_mapping_labels[np.argmax(logits)]
            results.append([is_trunc, orig_len, true_label_str, pred_label_str])
            
df_results = pd.DataFrame(results, columns=['is_truncated', 'original_length', 'true_labels', 'pred_labels'])

# Add analysis of truncation impact
truncated_count = df_results['is_truncated'].sum()
total_count = len(df_results)
print(f"Truncation analysis: {truncated_count}/{total_count} samples ({truncated_count/total_count*100:.1f}%) were truncated")

# Check accuracy for truncated vs non-truncated samples
if truncated_count > 0:
    truncated_acc = (df_results[df_results['is_truncated']]['true_labels'] == df_results[df_results['is_truncated']]['pred_labels']).mean()
    non_truncated_acc = (df_results[~df_results['is_truncated']]['true_labels'] == df_results[~df_results['is_truncated']]['pred_labels']).mean()
    print(f"Accuracy - Truncated: {truncated_acc*100:.1f}%, Non-truncated: {non_truncated_acc*100:.1f}%")

df_results.to_csv(config.OUTCSV, index=False)
print(f"Results saved to {config.OUTCSV}")