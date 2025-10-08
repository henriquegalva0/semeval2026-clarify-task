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

# Set default values for missing config attributes
if not hasattr(config, 'LEARNINGRATE'):
    config.LEARNINGRATE = 1e-5
if not hasattr(config, 'BATCHSIZE'):
    config.BATCHSIZE = 4
if not hasattr(config, 'NUMEPOCHS'):
    config.NUMEPOCHS = 3
if not hasattr(config, 'MAXLENGTH'):
    config.MAXLENGTH = 512
if not hasattr(config, 'MODELNAME'):
    config.MODELNAME = 'albert-base-v2'
if not hasattr(config, 'EXPERIMENTNAME'):
    config.EXPERIMENTNAME = "direct_clarity"  # or "evasion_based_clarity"
if not hasattr(config, 'OUTFILE'):
    config.OUTFILE = "./trained_model"

print(f"Using learning rate: {config.LEARNINGRATE}")
print(f"Using batch size: {config.BATCHSIZE}")
print(f"Using num epochs: {config.NUMEPOCHS}")
print(f"Using max length: {config.MAXLENGTH}")
print(f"Using model: {config.MODELNAME}")
print(f"Using experiment: {config.EXPERIMENTNAME}")

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

if config.EXPERIMENTNAME == "evasion_based_clarity":
    num_labels = 11
    mapping_labels = {'Explicit': 0, 'Implicit': 1, 'Dodging': 2, 'Deflection': 3, 'Partial/half-answer': 4, 'General': 5, 'Contradictory': 6, 'Declining to answer': 7, 'Claims ignorance': 8, 'Clarification': 9, 'Diffusion': 10}
elif config.EXPERIMENTNAME == "direct_clarity":
    num_labels = 3
    mapping_labels = {"Direct Reply": 0, "Indirect": 1, "Direct Non-Reply": 2}

from transformers import AlbertTokenizer, AlbertForSequenceClassification
tokenizer = AlbertTokenizer.from_pretrained(config.MODELNAME)
model = AlbertForSequenceClassification.from_pretrained(
    config.MODELNAME, 
    num_labels=num_labels
).to("cuda")

class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=config.MAXLENGTH):
        self.max_length = max_length
        self.texts, self.labels = [], []
        for text, label in zip(texts, labels):
            processed_text = self.smart_truncate(text, max_length)
            self.texts.append(processed_text)
            self.labels.append(label)
    
    def smart_truncate(self, text, max_tokens):
        """Smart truncation that preserves question and key parts of answer"""

        inputs = tokenizer(text, truncation=False, return_tensors='pt')
        token_count = len(inputs['input_ids'][0])
        
        if token_count <= max_tokens:
            return text
        
        parts = text.split("\n\nAnswer: ")
        if len(parts) < 2:
            truncated_tokens = inputs['input_ids'][0][:max_tokens-10] 
            return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        question_part = parts[0]
        answer_part = parts[1]
        
        if "\n\nSubanswer: " in answer_part:
            answer_parts = answer_part.split("\n\nSubanswer: ")
            main_answer = answer_parts[0]
            subanswer = answer_parts[1] if len(answer_parts) > 1 else ""
        else:
            main_answer = answer_part
            subanswer = ""
        
        question_tokens = tokenizer(question_part, return_tensors='pt')['input_ids'][0]
        
        available_tokens = max_tokens - len(question_tokens) - 20  # increased buffer
        
        if available_tokens <= 50:  
            truncated_tokens = question_tokens[:max_tokens-10]
            return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        answer_tokens = tokenizer(main_answer, return_tensors='pt')['input_ids'][0]
        
        if len(answer_tokens) > available_tokens:
            keep_start = available_tokens * 4 // 5  # 80% from start
            keep_end = available_tokens - keep_start  # 20% from end
            
            keep_start = min(keep_start, len(answer_tokens))
            keep_end = min(keep_end, len(answer_tokens))
            
            truncated_start = answer_tokens[:keep_start]
            truncated_end = answer_tokens[-keep_end:] if keep_end > 0 else torch.tensor([])
            
            if len(truncated_end) > 0:
                truncated_answer = tokenizer.decode(truncated_start, skip_special_tokens=True) + " ... " + tokenizer.decode(truncated_end, skip_special_tokens=True)
            else:
                truncated_answer = tokenizer.decode(truncated_start, skip_special_tokens=True) + " ..."
            
            final_text = f"{question_part}\n\nAnswer: {truncated_answer}"
            if subanswer:
                sub_tokens = tokenizer(subanswer, return_tensors='pt')['input_ids'][0]
                if len(sub_tokens) > 50:  # Reduced limit for subanswer
                    subanswer = tokenizer.decode(sub_tokens[:50], skip_special_tokens=True) + "..."
                final_text += f"\n\nSubanswer: {subanswer}"
            
            return final_text
        
        final_text = f"{question_part}\n\nAnswer: {main_answer}"
        if subanswer:
            final_text += f"\n\nSubanswer: {subanswer}"
        
        return final_text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(
            self.texts[idx],
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True
        )
        
        if len(inputs['input_ids'][0]) > self.max_length:
            inputs = tokenizer(
                self.texts[idx],
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
                stride=0  
            )
        
        label = torch.tensor(self.labels[idx])
        return inputs, label

# Load dataset from CSV file
dataset_path = "QAEvasion.csv"
df = pd.read_csv(dataset_path)

print(f"Dataset loaded with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

all_texts = []
all_labels = []

for index, row in df.iterrows():
    label_str = str(row["label"]).lower() if pd.notna(row["label"]) else ""
    if "other" not in label_str and pd.notna(row["label"]):
        text = f"Question: {row['interview_question']}\n\nAnswer: {row['interview_answer']}"
        
        if 'question' in df.columns and pd.notna(row['question']):
            text += f"\n\nSubanswer: {row['question']}"
        
        all_texts.append(text)
        
        if config.EXPERIMENTNAME == "evasion_based_clarity":
            all_labels.append(mapping_labels[row["label"]])
        elif config.EXPERIMENTNAME == "direct_clarity":
            all_labels.append(mapping_labels[clarity_mapping[row["label"]]])

print(f"Labels distribution: {pd.Series(all_labels).value_counts().sort_index()}")
print(f"Number of samples after filtering: {len(all_texts)}")

token_lengths = []
for text in all_texts[:10]:  # Check first 10 samples
    tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
    token_lengths.append(len(tokens))

print(f"Sample token lengths - Min: {min(token_lengths)}, Max: {max(token_lengths)}, Avg: {sum(token_lengths)/len(token_lengths):.1f}")

# Split data (adjust split ratio as needed)
split_idx = int(0.8 * len(all_texts))  # 80% train, 20% validation
train_texts, val_texts = all_texts[:split_idx], all_texts[split_idx:]
train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

# Create datasets and dataloaders
train_dataset = CustomDataset(train_texts, train_labels, max_length=config.MAXLENGTH)
val_dataset = CustomDataset(val_texts, val_labels, max_length=config.MAXLENGTH)

def collate_fn(batch):
    # Filter out None values and invalid items
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if len(batch) == 0:
        return None
    
    inputs, labels = zip(*batch)
    
    # Ensure all input tensors have the correct shape
    input_ids = torch.stack([x['input_ids'].squeeze() for x in inputs])
    attention_mask = torch.stack([x['attention_mask'].squeeze() for x in inputs])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': torch.stack(labels)
    }

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCHSIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCHSIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

# Use PyTorch's AdamW optimizer
optimizer = AdamW(model.parameters(), lr=config.LEARNINGRATE)

for epoch in range(config.NUMEPOCHS):
    # Training
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.NUMEPOCHS} - Training'):
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
        
        # Training accuracy
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total if train_total > 0 else 0
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{config.NUMEPOCHS} - Validation'):
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
    print(f'Epoch {epoch + 1}/{config.NUMEPOCHS}')
    print(f'Training Loss: {avg_train_loss:.4f} - Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Validation Loss: {average_val_loss:.4f} - Validation Accuracy: {accuracy * 100:.2f}%')
    print('-' * 50)

model.save_pretrained(config.OUTFILE)
tokenizer.save_pretrained(config.OUTFILE)
print(f"Model saved to {config.OUTFILE}")