import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import BertTokenizerFast

# Custom Transformer Encoder-Decoder Model
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_size, num_layers, max_length):
        super(CustomTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, ff_size, batch_first=True),
            num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, ff_size, batch_first=True),
            num_layers
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Adjust sequence lengths dynamically
        src = src[:, :MAX_LENGTH]
        tgt = tgt[:, :MAX_LENGTH]
        
        # Positional encoding
        src_positions = torch.arange(0, src.size(1)).unsqueeze(0).to(src.device)
        tgt_positions = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)

        src_embedding = self.token_embedding(src) + self.position_embedding(src_positions)
        tgt_embedding = self.token_embedding(tgt) + self.position_embedding(tgt_positions)

        src_embedding = self.dropout(src_embedding)
        tgt_embedding = self.dropout(tgt_embedding)

        memory = self.encoder(src_embedding, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)

        return self.fc_out(output)

# Hyperparameters
VOCAB_SIZE = 30522  # Vocabulary size (e.g., BERT tokenizer vocab size)
EMBED_SIZE = 512
NUM_HEADS = 8
FF_SIZE = 2048
NUM_LAYERS = 6
MAX_LENGTH = 256

# Initialize the model
custom_model = CustomTransformer(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, FF_SIZE, NUM_LAYERS, MAX_LENGTH)

# Dataset Class
class RealSummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=256):
        self.tokenizer = tokenizer
        self.inputs, self.targets = self._load_dataset(data_path)
        self.max_length = max_length

    def _load_dataset(self, path):
        inputs = []
        targets = []
        judgement_path = os.path.join(path, "train-data", "judgement")
        summary_path = os.path.join(path, "train-data", "summary")

        for fname in os.listdir(judgement_path):
            with open(os.path.join(judgement_path, fname), 'r', encoding='utf-8') as f:
                inputs.append(f.read())
            with open(os.path.join(summary_path, fname), 'r', encoding='utf-8') as f:
                targets.append(f.read())

        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(self.inputs[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_enc = self.tokenizer(self.targets[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return input_enc['input_ids'].squeeze(), target_enc['input_ids'].squeeze()

# Dataset Path
DATASET_PATH = r"C:\\Users\\Ramachandra\\OneDrive\\Desktop\\FYP\\dataset (3)\\dataset\\IN-Abs"

# Load tokenizer and dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
real_dataset = RealSummarizationDataset(tokenizer, DATASET_PATH)

# Create DataLoader
real_dataloader = DataLoader(real_dataset, batch_size=2, shuffle=True)

# Training and Accuracy Calculation
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(custom_model.parameters(), lr=5e-5)

epochs = 3
total_correct = 0
total_tokens = 0

for epoch in range(epochs):
    custom_model.train()
    for src, tgt in real_dataloader:
        src, tgt = src.to("cpu"), tgt.to("cpu")
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        outputs = custom_model(src, tgt_input)
        outputs = outputs[:, :tgt_output.size(1), :]

        # Compute loss
        loss = criterion(outputs.view(-1, VOCAB_SIZE), tgt_output.view(-1))

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = tgt_output != tokenizer.pad_token_id
        correct = (predictions == tgt_output) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Final Accuracy
accuracy = (total_correct / total_tokens) * 100
print(f"Final Accuracy: {accuracy:.2f}%")
