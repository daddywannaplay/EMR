"""
Test Full-Text OCR Model - Generate Complete Text Files for Each Test Image
Output: For each test image, create a text file with predicted full text like 1.jpg → 1.txt
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from train_model import CRNN


class OCRDataset(Dataset):
    """Load dataset with full text content"""
    
    def __init__(self, data_path, doc_type, split='test'):
        self.image_dir = f"{data_path}/{doc_type}/{split}"
        self.samples = []
        self.vocab = set()
        
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
                                self.samples.append((img_path, text, f))
                                self.vocab.update(text)
                        except:
                            pass
        
        self.vocab = sorted(list(self.vocab))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text, filename = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize((256, 64))
            image = torch.FloatTensor(np.array(image)) / 255.0
            
            # Convert text to indices and pad to fixed length
            text_indices = [self.char2idx.get(c, 0) for c in text[:256]]
            # Pad to 256 characters
            while len(text_indices) < 256:
                text_indices.append(0)
            text_indices = torch.tensor(text_indices[:256])
            
            return image, text_indices, text, filename
        except:
            return None, None, None, None


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        
        from torchvision import models
        resnet50 = models.resnet50(pretrained=True)
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(TextDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128 + 256, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, encoder_output, text_indices):
        encoder_output = encoder_output.unsqueeze(1).expand(-1, text_indices.size(1), -1)
        text_embed = self.embedding(text_indices)
        decoder_input = torch.cat([text_embed, encoder_output], dim=2)
        lstm_output, _ = self.lstm(decoder_input)
        output = self.fc(lstm_output)
        return output


class FullTextOCRModel(nn.Module):
    def __init__(self, vocab_size):
        super(FullTextOCRModel, self).__init__()
        self.encoder = TextEncoder()
        self.decoder = TextDecoder(vocab_size)
    
    def forward(self, images, text_indices):
        encoder_output = self.encoder(images)
        output = self.decoder(encoder_output, text_indices)
        return output


def test_ocr_model(data_path, model_path, vocab_path, doc_type, output_dir):
    """Test model and generate full text predictions for each image"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"TESTING {doc_type.upper()} MODEL - FULL TEXT GENERATION")
    print(f"{'='*70}")
    print(f"Device: {device}\n")
    
    # Load dataset and vocabulary
    test_dataset = OCRDataset(data_path, doc_type, 'test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False)
    
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = FullTextOCRModel(vocab_data['vocab_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded\n")
    
    # Create output directory for text predictions
    pred_dir = f"{output_dir}/{doc_type}_predicted_text"
    os.makedirs(pred_dir, exist_ok=True)
    
    print(f"Generating full text predictions...\n")
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, texts, actual_texts, filenames = batch
            if images is None or images.shape[0] == 0:
                continue
            
            images = images.unsqueeze(1).to(device)
            
            # Get predictions
            outputs = model(images, texts)
            predicted_indices = outputs.argmax(dim=2)
            
            # Convert to text
            for pred_indices, actual_text, filename in zip(predicted_indices, actual_texts, filenames):
                # Convert indices to text
                gen_text = ''.join([idx2char.get(int(idx.item()), '') for idx in pred_indices])
                
                # Save to text file (just like 1.jpg → 1.txt)
                base_name = filename.rsplit('.', 1)[0]
                pred_file = os.path.join(pred_dir, base_name + '.txt')
                with open(pred_file, 'w', encoding='utf-8') as f:
                    f.write(gen_text)
                
                # Check if matches
                is_correct = gen_text.strip() == actual_text.strip()
                
                print(f"  {filename:40s} → {base_name}.txt {'✓' if is_correct else '✗'}")
                
                # Store result
                all_results.append({
                    'input_image': filename,
                    'output_text_file': pred_file,
                    'generated_text': gen_text,
                    'actual_text': actual_text,
                    'matches': is_correct
                })
            
            print(f"Batch {batch_idx + 1}/{len(test_loader)}\n")
    
    # Calculate accuracy
    correct = sum(1 for r in all_results if r['matches'])
    accuracy = (correct / len(all_results)) * 100 if all_results else 0
    
    # Save summary
    summary = {
        'model_type': doc_type,
        'total_test_samples': len(all_results),
        'correct_predictions': correct,
        'accuracy': f"{accuracy:.2f}%",
        'output_directory': pred_dir,
        'model_path': model_path,
        'description': 'Each image generates a text file with predicted content'
    }
    
    summary_file = f"{output_dir}/{doc_type}_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    results_file = f"{output_dir}/{doc_type}_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print(f"✓ TESTING COMPLETE FOR {doc_type.upper()}")
    print("="*70)
    print(f"\n📊 RESULTS:")
    print(f"  Total Test Images: {len(all_results)}")
    print(f"  Correct Predictions: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"\n📁 OUTPUT TEXT FILES:")
    print(f"  Location: {pred_dir}/")
    print(f"  Files: {len(all_results)} text files (one per image)")
    print(f"  Format: image.jpg → image.txt (with predicted content)")
    print(f"\n📄 REPORTS:")
    print(f"  Summary: {summary_file}")
    print(f"  Details: {results_file}\n")
    
    return summary, all_results


def infer_ocr_model(data_path, model_path, vocab_path, doc_type, output_dir, max_len=256):
    """Inference-only: generate text files from images without requiring ground-truth .txt files.
    Uses <SOS> and stops on <EOS> from the saved vocabulary."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"INFERENCE {doc_type.upper()} MODEL - IMAGE → TEXT (NO .TXT REQUIRED)")
    print(f"{'='*70}")
    print(f"Device: {device}\n")

    img_dir = f"{data_path}/{doc_type}/test"
    if not os.path.exists(img_dir):
        print(f"No test folder found at {img_dir}")
        return {}, []

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    idx2char = {int(k): v for k, v in vocab_data.get('idx2char', {}).items()} if 'idx2char' in vocab_data else {}
    char2idx = vocab_data.get('char2idx', {})

    sos_idx = char2idx.get('<SOS>', 1)
    eos_idx = char2idx.get('<EOS>', 2)
    vocab_size = vocab_data.get('vocab_size', len(idx2char) or 0)

    # Load model
    print(f"Loading model: {model_path}")
    model = FullTextOCRModel(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded\n")

    pred_dir = f"{output_dir}/{doc_type}_predicted_text"
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    image_files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(img_dir, filename)
            try:
                image = Image.open(img_path).convert('L')
                image = image.resize((256, 64))
                image = torch.FloatTensor(np.array(image)) / 255.0
                image = image.unsqueeze(0).unsqueeze(1).to(device)  # [1,1,H,W]

                encoder_output = model.encoder(image)

                # init hidden
                num_layers = model.decoder.lstm.num_layers
                hidden_size = model.decoder.lstm.hidden_size
                h = torch.zeros(num_layers, 1, hidden_size, device=device)
                c = torch.zeros(num_layers, 1, hidden_size, device=device)

                prev_idx = torch.tensor([sos_idx], dtype=torch.long, device=device)
                gen_indices = []

                for t in range(max_len):
                    emb = model.decoder.embedding(prev_idx).unsqueeze(1)
                    enc_exp = encoder_output.unsqueeze(1)
                    dec_in = torch.cat([emb, enc_exp], dim=2)
                    out, (h, c) = model.decoder.lstm(dec_in, (h, c))
                    logits = model.decoder.fc(out.squeeze(1))
                    next_idx = logits.argmax(dim=1)
                    idx_val = int(next_idx.item())
                    if idx_val == eos_idx:
                        break
                    gen_indices.append(idx_val)
                    prev_idx = next_idx

                gen_text = ''.join([idx2char.get(i, '') for i in gen_indices])

                base_name = filename.rsplit('.', 1)[0]
                pred_file = os.path.join(pred_dir, base_name + '.txt')
                with open(pred_file, 'w', encoding='utf-8') as f:
                    f.write(gen_text)

                print(f"  {filename:40s} → {base_name}.txt")
                results.append({'input_image': filename, 'output_text_file': pred_file, 'generated_text': gen_text})

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    summary = {
        'model_type': doc_type,
        'total_inferred': len(results),
        'output_directory': pred_dir,
        'model_path': model_path,
        'description': 'Inference-only: images → predicted text files'
    }
    summary_file = f"{output_dir}/{doc_type}_inference_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '='*70)
    print(f"✓ INFERENCE COMPLETE FOR {doc_type.upper()}")
    print('='*70)
    print(f"  Total Inferred: {len(results)}")
    print(f"  Outputs: {pred_dir}/")

    return summary, results


def infer_crnn_model(data_path, model_path, vocab_path, doc_type, output_dir):
    """Run CRNN model + greedy CTC decoding on images (no .txt required)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"CRNN CTC INFERENCE {doc_type.upper()} - IMAGE → TEXT")
    print(f"{'='*70}")
    print(f"Device: {device}\n")

    img_dir = f"{data_path}/{doc_type}/test"
    if not os.path.exists(img_dir):
        print(f"No test folder: {img_dir}")
        return {}, []

    # load vocab
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    idx2char = {int(k): v for k, v in vocab_data.get('idx2char', {}).items()} if 'idx2char' in vocab_data else {}
    vocab_size = vocab_data.get('vocab_size', len(idx2char) or 0)
    blank_idx = 0

    # load model
    print(f"Loading CRNN: {model_path}")
    model = CRNN(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pred_dir = f"{output_dir}/{doc_type}_predicted_text"
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    image_files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(img_dir, filename)
            try:
                image = Image.open(img_path).convert('L')
                image = image.resize((256, 64))
                img_t = torch.FloatTensor(np.array(image)) / 255.0
                img_t = img_t.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

                logits = model(img_t)  # [T, B, C]
                preds = logits.argmax(dim=2).squeeze(1).cpu().numpy().tolist()  # [T]

                # Collapse repeats and remove blanks
                collapsed = []
                prev = None
                for p in preds:
                    if p == prev:
                        prev = p
                        continue
                    if p == blank_idx:
                        prev = p
                        continue
                    collapsed.append(p)
                    prev = p

                gen_text = ''.join([idx2char.get(i, '') for i in collapsed])

                base = filename.rsplit('.', 1)[0]
                out_file = os.path.join(pred_dir, base + '.txt')
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(gen_text)

                print(f"  {filename:40s} → {base}.txt")
                results.append({'input_image': filename, 'output_text_file': out_file, 'generated_text': gen_text})
            except Exception as e:
                print(f"Error {filename}: {e}")

    summary = {'model_type': doc_type, 'total_inferred': len(results), 'output_directory': pred_dir, 'model_path': model_path}
    summary_file = f"{output_dir}/{doc_type}_crnn_inference_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '='*70)
    print(f"✓ CRNN INFERENCE COMPLETE FOR {doc_type.upper()}")
    print('='*70)
    print(f"  Total Inferred: {len(results)}")
    print(f"  Outputs: {pred_dir}/")

    return summary, results


if __name__ == "__main__":
    try:
        from google.colab import drive
        # In Colab, data is directly accessible
        data_path = "split_data"
        model_dir = "ocr_models"
        output_dir = "ocr_models"
    except ImportError:
        # Local machine
        data_path = "split_data"
        model_dir = "ocr_models"
        output_dir = "ocr_models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FULL-TEXT OCR MODEL TESTING")
    print(f"{'='*70}")
    
    # Test prescriptions
    prescr_model_path = f"{model_dir}/prescriptions_best_model.pt"
    prescr_vocab_path = f"{model_dir}/prescriptions_vocab.json"
    
    if os.path.exists(prescr_model_path) and os.path.exists(prescr_vocab_path):
        infer_crnn_model(data_path, prescr_model_path, prescr_vocab_path, 'prescriptions', output_dir)
    else:
        print(f"✗ Prescription model or vocab not found")
    
    # Test lab reports
    lab_model_path = f"{model_dir}/lab_reports_best_model.pt"
    lab_vocab_path = f"{model_dir}/lab_reports_vocab.json"
    
    if os.path.exists(lab_model_path) and os.path.exists(lab_vocab_path):
        infer_crnn_model(data_path, lab_model_path, lab_vocab_path, 'lab_reports', output_dir)
    else:
        print(f"✗ Lab reports model or vocab not found")
    
    print(f"\n{'='*70}")
    print("✓ ALL TESTS COMPLETED!")
    print(f"{'='*70}\n")
