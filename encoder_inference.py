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
    def __init__(self, texts, labels, max_length=config.MAXLENGTH):  # You can set max_length to an appropriate value

        self.max_length = max_length
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            # truncation=True,
            padding='max_length',  # Use padding to ensure all sequences have the same length
            max_length=self.max_length
        )

        is_truncated = False
        if len(inputs['input_ids'][0]) > self.max_length:
            is_truncated = True

        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            truncation=True,
            padding='max_length',  # Use padding to ensure all sequences have the same length
            max_length=self.max_length
        )
           
        label = torch.tensor(self.labels[idx])
        return inputs, label, is_truncated

def collate_fn(batch):
    inputs, labels, is_truncated = zip(*batch)
    return {
        'input_ids': torch.stack([x['input_ids'].squeeze() for x in inputs]),
        'attention_mask': torch.stack([x['attention_mask'].squeeze() for x in inputs]),
        'labels': torch.tensor(labels), 
        'is_truncated': is_truncated
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
max_size = config.MAXSIZE 

labels = []

for _, row in config.TESTINGDATASET.iterrows():
    l = [row["Annotator1"], row["Annotator2"], row["Annotator3"]]
    labels.append(max(set(l), key=labels.count))
config.TESTINGDATASET["Label"] = labels

all_texts = [f"Question: {row['Interview Question']}\n\nAnswer: {row['Interview Answer']}\n\nSubanswer: {row['Question']}" for _, row in config.TESTINGDATASET.iterrows()] # * problema dos tokens excedentes * (tentar trocar question pela PRIMEIRA PARTE do summary do gpt)

if config.EXPERIMENTNAME == "evasion_based_clarity":
    all_labels = [mapping_labels[row["Label"]] for _, row in config.TESTINGDATASET.iterrows() if "other" not in row["Label"].lower()]
elif config.EXPERIMENTNAME == "direct_clarity":
    all_labels = [mapping_labels[clarity_mapping[row["Label"]]] for _, row in config.TESTINGDATASET.iterrows() if "other" not in row["Label"].lower()]

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
        
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        for true_label, pred_label, is_trunc in zip(labels.cpu().numpy(), outputs["logits"].cpu().numpy(), is_truncated):
            true_label = inv_mapping_labels[true_label]
            pred_label = inv_mapping_labels[np.argmax(pred_label)]
            results.append([is_trunc, true_label, pred_label])
            
df = pd.DataFrame(results, columns=['is_truncated', 'true_labels', 'pred_labels'])
df.to_csv(config.OUTCSV)
