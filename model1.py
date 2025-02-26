import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Utility functions and classes from BART_utilities.py
# -- START OF UTILITY FUNCTIONS --
def freeze_params(model):
    for layer in model.parameters():
        layer.requires_grad = False

def shift_tokens_right(input_ids, pad_token_id):
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=1024, min_length=512):
    input_ids, attention_masks, target_ids = [], [], []

    for sentence in source_sentences:
        encoded_dict = tokenizer(
            sentence, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    for sentence in target_sentences:
        encoded_dict = tokenizer(
            sentence, max_length=min_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        target_ids.append(encoded_dict['input_ids'])

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_masks, dim=0),
        "labels": torch.cat(target_ids, dim=0),
    }

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        src_ids, src_mask, tgt_ids = batch[0], batch[1], batch[2]
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
            outputs.logits.view(-1, outputs.logits.size(-1)), tgt_ids.view(-1)
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        src_ids, src_mask, tgt_ids = batch[0], batch[1], batch[2]
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        val_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
            outputs.logits.view(-1, outputs.logits.size(-1)), tgt_ids.view(-1)
        )
        return {'val_loss': val_loss}

    def generate_text(self, text, eval_beams=4, max_len=1024, early_stopping=True):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True)
        generated_ids = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=eval_beams,
            max_length=max_len,
            early_stopping=early_stopping
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

# -- END OF UTILITY FUNCTIONS --

# Dataset Loading Function
def load_dataset(path):
    train_path = os.path.join(path, "train-data")
    test_path = os.path.join(path, "test-data")
    
    # Assuming data is in text format; modify as needed
    train_src, train_tgt = [], []
    test_src, test_tgt = [], []

    for fname in os.listdir(os.path.join(train_path, "judgement")):
        with open(os.path.join(train_path, "judgement", fname), 'r') as f:
            train_src.append(f.read())
        with open(os.path.join(train_path, "summary", fname), 'r') as f:
            train_tgt.append(f.read())

    for fname in os.listdir(os.path.join(test_path, "judgement")):
        with open(os.path.join(test_path, "judgement", fname), 'r') as f:
            test_src.append(f.read())
        with open(os.path.join(test_path, "summary", fname), 'r') as f:
            test_tgt.append(f.read())

    return train_src, train_tgt, test_src, test_tgt

# Main script
if __name__ == "__main__":
    DATASET_PATH = r"C:\\Users\\Ramachandra\\OneDrive\\Desktop\\FYP\\dataset (3)\\dataset\\IN-Abs"
    MODEL_SAVE_PATH = "fine_tuned_bart_model.pt"

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    # Check if fine-tuned model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print("Loading fine-tuned model...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        print("Fine-tuning model...")
        train_src, train_tgt, test_src, test_tgt = load_dataset(DATASET_PATH)

        train_data = encode_sentences(tokenizer, train_src, train_tgt)

        train_dataset = TensorDataset(
            train_data['input_ids'], train_data['attention_mask'], train_data['labels']
        )
        train_loader = DataLoader(train_dataset, batch_size=2, sampler=RandomSampler(train_dataset))

        lit_model = LitModel(learning_rate=3e-5, tokenizer=tokenizer, model=model)

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator='cpu',  # Use CPU only
            callbacks=[ModelCheckpoint(dirpath=".", filename="best_model")]
        )
        trainer.fit(lit_model, train_loader)

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Generate a summary
    lit_model = LitModel(learning_rate=3e-5, tokenizer=tokenizer, model=model)

    # Load an example document from the test dataset
    example_document_path = os.path.join(DATASET_PATH, "test-data", "judgement")
    example_file = next(iter(os.listdir(example_document_path)))  # Pick the first file
    with open(os.path.join(example_document_path, example_file), 'r') as f:
        example_text = f.read()

    summary = lit_model.generate_text(example_text)
    print("Generated Summary:", summary)
