"""
Train Full-Text OCR Model - Generate Complete Text from Images
Sequence-to-Sequence based model for prescription and lab report text recognition
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F


class OCRDataset(Dataset):
    """CTC-friendly dataset: returns image tensor and label indices (no SOS/EOS).

    Vocabulary uses blank token at index 0 for CTC.
    """

    def __init__(self, data_path, doc_type, split='train'):
        self.image_dir = f"{data_path}/{doc_type}/{split}"
        self.samples = []
        self.chars = set()

        if os.path.exists(self.image_dir):
            for f in sorted(os.listdir(self.image_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.image_dir, f)
                    txt_name = f.rsplit('.', 1)[0] + '.txt'
                    txt_path = os.path.join(self.image_dir, txt_name)
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                                text = file.read().strip()
                            if text:
                                self.samples.append((img_path, text))
                                self.chars.update(text)
                        except:
                            pass

        # CTC blank token at index 0
        base_chars = sorted(list(self.chars))
        all_chars = ['<BLANK>'] + base_chars
        self.char2idx = {c: i for i, c in enumerate(all_chars)}
        self.idx2char = {i: c for i, c in enumerate(all_chars)}
        self.vocab_size = len(all_chars)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize((256, 64))
            image = torch.FloatTensor(np.array(image)) / 255.0
            # labels: convert to indices (no blanks, no SOS/EOS)
            label = [self.char2idx.get(c, 0) for c in text]
            label = torch.tensor(label, dtype=torch.long)
            return image, label, text
        except:
            return None, None, None


def ctc_collate(batch):
    """Collate function that stacks images and concatenates labels for CTC.

    Returns:
      images: Tensor [B,1,H,W]
      labels: 1D LongTensor concatenated
      label_lengths: LongTensor [B]
      input_width: int (encoder time steps after conv)
      raw_texts: list of strings
    """
    batch = [b for b in batch if b[0] is not None]
    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    texts = [b[2] for b in batch]

    images = torch.stack([img.unsqueeze(0) for img in images], dim=0)  # [B,1,H,W]

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    if len(labels) > 0:
        labels_cat = torch.cat(labels)
    else:
        labels_cat = torch.tensor([], dtype=torch.long)

    return images, labels_cat, label_lengths, texts


class CRNN(nn.Module):
    """CRNN: small CNN -> BiLSTM -> Linear (for CTC)

    Produces logits of shape (T, B, vocab_size) where T is time steps.
    """
    def __init__(self, vocab_size, imgH=64, nc=1, nh=256, n_rnn=2, out_w=32):
        super(CRNN, self).__init__()
        self.vocab_size = vocab_size
        self.out_w = out_w

        # simple conv layers
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True)
        )

        # After convs we'll have a feature map [B, C, 1, W'] where W' depends on input
        # Project conv channels to rnn hidden size
        rnn_input_size = 512

        self.rnn = nn.LSTM(rnn_input_size, nh, num_layers=n_rnn, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nh * 2, vocab_size)

    def forward(self, x):
        # x: [B,1,H,W]
        conv = self.cnn(x)  # [B, C, Hc, Wc]
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)  # [B, C, W]
        conv = conv.permute(0, 2, 1)  # [B, W, C]

        # rnn expects [B, T, C]
        rnn_out, _ = self.rnn(conv)
        # rnn_out: [B, T, 2*nh]
        logits = self.embedding(rnn_out)  # [B, T, vocab_size]
        # transpose to (T, B, C) for CTCLoss
        logits = logits.permute(1, 0, 2)
        return logits


def train_ocr_model(data_path, doc_type, output_dir, epochs=20, batch_size=8):
    """Train full-text OCR model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Full-Text OCR Model - {doc_type.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}\n")
    
    # Load datasets
    train_dataset = OCRDataset(data_path, doc_type, 'train')
    val_dataset = OCRDataset(data_path, doc_type, 'validation')

    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    # Use collate_fn for CTC
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=ctc_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, collate_fn=ctc_collate)

    # Model
    model = CRNN(train_dataset.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    # Training
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images, labels_cat, label_lengths, texts = batch
            if images is None or images.shape[0] == 0:
                continue

            images = images.to(device)
            labels_cat = labels_cat.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()
            logits = model(images)  # [T, B, C]

            # input_lengths: T per sample (all equal here)
            T = logits.size(0)
            batch_size_local = logits.size(1)
            input_lengths = torch.full((batch_size_local,), T, dtype=torch.long, device=device)

            log_probs = F.log_softmax(logits, dim=2)
            loss = criterion(log_probs, labels_cat, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
        
        if train_batches == 0:
            print(f"Warning: No valid training batches in epoch {epoch+1}")
            continue
        
        avg_train_loss = train_loss / train_batches
        
        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels_cat, label_lengths, texts in val_loader:
                if images is None or images.shape[0] == 0:
                    continue

                images = images.to(device)
                labels_cat = labels_cat.to(device)
                label_lengths = label_lengths.to(device)

                logits = model(images)
                T = logits.size(0)
                batch_size_local = logits.size(1)
                input_lengths = torch.full((batch_size_local,), T, dtype=torch.long, device=device)

                log_probs = F.log_softmax(logits, dim=2)
                loss = criterion(log_probs, labels_cat, input_lengths, label_lengths)
                val_loss += loss.item()
                val_batches += 1
        
        if val_batches == 0:
            print(f"Warning: No valid validation batches in epoch {epoch+1}")
            avg_val_loss = best_val_loss
        else:
            avg_val_loss = val_loss / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/{doc_type}_best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}\n")
                break
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/{doc_type}_model.pt")
    
    # Save history
    with open(f"{output_dir}/{doc_type}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save vocabulary
    vocab_data = {
        'char2idx': train_dataset.char2idx,
        'idx2char': {str(k): v for k, v in train_dataset.idx2char.items()},
        'vocab_size': train_dataset.vocab_size
    }
    with open(f"{output_dir}/{doc_type}_vocab.json", 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETE - {doc_type.upper()}")
    print(f"{'='*70}\n")
    
    return model, train_dataset, history


if __name__ == "__main__":
    try:
        from google.colab import drive
        # In Colab, data is directly accessible
        data_path = "split_data"
        output_dir = "ocr_models"
    except ImportError:
        # Local machine
        data_path = "split_data"
        output_dir = "ocr_models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FULL-TEXT OCR MODEL TRAINING")
    print(f"{'='*70}")
    
    # Train prescriptions
    prescr_model, prescr_dataset, prescr_history = train_ocr_model(data_path, 'prescriptions', output_dir, epochs=20)
    
    # Train lab reports
    lab_model, lab_dataset, lab_history = train_ocr_model(data_path, 'lab_reports', output_dir, epochs=20)
    
    print(f"\n{'='*70}")
    print("✓ ALL MODELS TRAINED!")
    print(f"{'='*70}\n")
