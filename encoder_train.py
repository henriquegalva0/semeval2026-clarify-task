import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW  # Import from torch.optim instead

import argparse
import pandas as pd
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

import torch
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='')
    
# Adding arguments
parser.add_argument('--model_name', type=str)
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

model_name = args.model_name
experiment = args.experiment

clarity_mapping ={
    'Explicit': 'Direct Reply',
    'Implicit': 'Indirect',
    'Dodging': "Indirect",
    'Deflection': "Indirect",
    'Partial/half-answer': "Indirect",
    'General': "Indirect",
    'Declining to answer': "Direct Non-Reply",
    'Claims ignorance': "Direct Non-Reply",
    'Clarification': "Direct Non-Reply",
}

if experiment == "evasion_based_clarity":
    num_labels = 11
    mapping_labels = {'Explicit': 0, 'Implicit': 1, 'Dodging': 2, 'Deflection': 3, 'Partial/half-answer': 4, 'General': 5, 'Contradictory': 6, 'Declining to answer': 7, 'Claims ignorance': 8, 'Clarification': 9, 'Diffusion': 10}
elif experiment == "direct_clarity":
    num_labels = 3
    mapping_labels = {"Direct Reply": 0, "Indirect": 1, "Direct Non-Reply": 2}

# Load pre-trained model and tokenizer
'''if 'roberta' in model_name: 
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_size = 512'''

if "xlnet" in model_name:
    from transformers import XLNetTokenizer, XLNetForSequenceClassification
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_size = 4096

elif "deberta" in model_name: 
    from transformers import DebertaTokenizer, DebertaForSequenceClassification
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    model = DebertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_size = 512

elif "modernbert" in model_name.lower():
    print("Loading ModernBERT with memory optimizations...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        device_map="auto",        
        trust_remote_code=True
    )
    
    max_size = 8192  
    batch_size = 1  

elif "albert" in model_name.lower():
    from transformers import AlbertTokenizer, AlbertForSequenceClassification
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    ).to("cuda")
    max_size = 512  # ALBERT normalmente segue BERT em limite de tokens

elif "distilroberta" in model_name.lower():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")
    max_length = 512    

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=512):
        self.max_length = max_length
        self.texts, self.labels = [], []
        for text, label in zip(texts, labels):
            inputs = tokenizer(text,
                             return_tensors='pt',
                             padding='max_length',
                             max_length=self.max_length)
            
            if len(inputs['input_ids'][0]) > self.max_length:
                continue
            self.texts.append(text)
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx],
                         return_tensors='pt',
                         padding='max_length',
                         max_length=self.max_length)

        if len(inputs['input_ids'][0]) > self.max_length:
            return None, None

        label = torch.tensor(self.labels[idx])
        return inputs, label

# Load and process data
dataset = pd.read_csv("./drive/MyDrive/Projetos/ICnadia[FinetuningNLP]/SemEval2026/Question-Evasion/dataset/QAEvasion.csv")

# Fix: Properly iterate through the DataFrame
all_texts = []
all_labels = []

for index, row in dataset.iterrows():
    label_str = str(row["label"]).lower() if pd.notna(row["label"]) else ""
    if "other" not in label_str:
        text = f"Question: {row['interview_question']}\n\nAnswer: {row['interview_answer']}\n\nSubanswer: {row['question']}"
        all_texts.append(text)
        
        if experiment == "evasion_based_clarity":
            all_labels.append(mapping_labels[row["label"]])
        elif experiment == "direct_clarity":
            all_labels.append(mapping_labels[clarity_mapping[row["label"]]])

print(f"Labels: {set(all_labels)}")
print(f"Number of samples: {len(all_texts)}")

# Split data
train_texts, val_texts, train_labels, val_labels = all_texts[:2700], all_texts[2700:], all_labels[:2700], all_labels[2700:]

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels, max_length=512)
val_dataset = CustomDataset(val_texts, val_labels, max_length=512)

def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    
    inputs, labels = zip(*batch)
    return {
        'input_ids': torch.stack([x['input_ids'].squeeze() for x in inputs]),
        'attention_mask': torch.stack([x['attention_mask'].squeeze() for x in inputs]),
        'labels': torch.stack(labels)
    }

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

# Use PyTorch's AdamW optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 5
for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
        if batch is None:
            continue
            
        inputs = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['labels'].to("cuda")
    
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            if batch is None:
                continue
                
            inputs = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")
    
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
    
    average_val_loss = val_loss / len(val_dataloader)
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {average_val_loss:.4f} - Accuracy: {accuracy * 100:.2f}%')

# Save the fine-tuned model
out_file = f"{model_name.split('/')[-1]}-qaevasion-{experiment}"
model.save_pretrained(out_file)
tokenizer.save_pretrained(out_file)  # Don't forget to save the tokenizer too!