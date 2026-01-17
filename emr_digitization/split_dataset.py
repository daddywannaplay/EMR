"""
Split Dataset: Prescriptions (data1) and Lab Reports (lbmaske)
into Train/Test/Validation sets with proper organization
"""

import os
import shutil
import random
from pathlib import Path
import json


def split_dataset(data_folder="data", output_folder="split_data", train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, seed=42):
    """
    Split prescriptions and lab reports into train/test/validation
    
    Output structure:
    split_data/
    ├── prescriptions/
    │   ├── train/ (images + txt)
    │   ├── test/ (images + txt)
    │   └── validation/ (images + txt)
    └── lab_reports/
        ├── train/ (images + txt)
        ├── test/ (images + txt)
        └── validation/ (images + txt)
    """
    
    random.seed(seed)
    
    # Create output directories
    for doc_type in ['prescriptions', 'lab_reports']:
        for split in ['train', 'test', 'validation']:
            os.makedirs(f"{output_folder}/{doc_type}/{split}", exist_ok=True)
    
    print("\n" + "="*70)
    print("SPLITTING DATASET INTO TRAIN/TEST/VALIDATION")
    print("="*70 + "\n")
    
    # Process Prescriptions (data1)
    print("Processing PRESCRIPTIONS (data1)...")
    prescr_input = f"{data_folder}/data1/Input"
    prescr_output = f"{data_folder}/data1/Output"
    
    print(f"  Looking in: {prescr_input}")
    print(f"  Looking in: {prescr_output}")
    
    prescr_samples = []
    if os.path.exists(prescr_input) and os.path.exists(prescr_output):
        for img_file in sorted(os.listdir(prescr_input)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(prescr_input, img_file)
                txt_file = img_file.rsplit('.', 1)[0] + '.txt'
                txt_path = os.path.join(prescr_output, txt_file)
                
                if os.path.exists(txt_path):
                    prescr_samples.append({'image': img_path, 'text': txt_path, 'name': img_file})
        
        print(f"✓ Found {len(prescr_samples)} prescription files\n")
    else:
        print(f"✗ ERROR: Prescription folders not found!")
        print(f"  Input exists: {os.path.exists(prescr_input)}")
        print(f"  Output exists: {os.path.exists(prescr_output)}")
        print(f"\n  Please check your data_folder path and run check_colab_paths.py to diagnose\n")
    
    # Process Lab Reports (lbmaske)
    print("Processing LAB REPORTS (lbmaske)...")
    lab_input = f"{data_folder}/lbmaske/Input"
    lab_output = f"{data_folder}/lbmaske/Output"
    
    print(f"  Looking in: {lab_input}")
    print(f"  Looking in: {lab_output}")
    
    lab_samples = []
    if os.path.exists(lab_input) and os.path.exists(lab_output):
        for img_file in sorted(os.listdir(lab_input)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(lab_input, img_file)
                txt_file = img_file.rsplit('.', 1)[0] + '.txt'
                txt_path = os.path.join(lab_output, txt_file)
                
                if os.path.exists(txt_path):
                    lab_samples.append({'image': img_path, 'text': txt_path, 'name': img_file})
        
        print(f"✓ Found {len(lab_samples)} lab report files\n")
    else:
        print(f"✗ ERROR: Lab report folders not found!")
        print(f"  Input exists: {os.path.exists(lab_input)}")
        print(f"  Output exists: {os.path.exists(lab_output)}")
        print(f"\n  Please check your data_folder path and run check_colab_paths.py to diagnose\n")
    
    # Split function
    def split_samples(samples, train_ratio, test_ratio, val_ratio):
        random.shuffle(samples)
        total = len(samples)
        train_size = int(total * train_ratio)
        test_size = int(total * test_ratio)
        
        return {
            'train': samples[:train_size],
            'test': samples[train_size:train_size + test_size],
            'validation': samples[train_size + test_size:]
        }
    
    # Check if we have data to process
    if len(prescr_samples) == 0 and len(lab_samples) == 0:
        print("\n" + "="*70)
        print("✗ ERROR: NO DATA FOUND!")
        print("="*70)
        print("\nNo prescription or lab report files were found.")
        print("\nTo fix this:")
        print("1. In Colab, upload your data folder to Google Drive")
        print("2. The structure should be:")
        print("   MyDrive/")
        print("   └── data/  (or your custom folder name)")
        print("       ├── data1/")
        print("       │   ├── Input/  (JPG images)")
        print("       │   └── Output/ (TXT files)")
        print("       └── lbmaske/")
        print("           ├── Input/  (PNG images)")
        print("           └── Output/ (TXT files)")
        print("\n3. Run: !python check_colab_paths.py")
        print("4. Find the correct path to your data")
        print("5. Update the data_folder variable in split_dataset.py")
        print("="*70 + "\n")
        return None
    
    # Split prescriptions
    prescr_split = split_samples(prescr_samples, train_ratio, test_ratio, val_ratio)
    
    # Split lab reports
    lab_split = split_samples(lab_samples, train_ratio, test_ratio, val_ratio)
    
    # Copy files
    def copy_split(split_dict, doc_type, output_folder):
        for split_name, samples in split_dict.items():
            for sample in samples:
                img_dest = f"{output_folder}/{doc_type}/{split_name}/{sample['name']}"
                txt_name = sample['name'].rsplit('.', 1)[0] + '.txt'
                txt_dest = f"{output_folder}/{doc_type}/{split_name}/{txt_name}"
                
                shutil.copy2(sample['image'], img_dest)
                shutil.copy2(sample['text'], txt_dest)
    
    copy_split(prescr_split, 'prescriptions', output_folder)
    copy_split(lab_split, 'lab_reports', output_folder)
    
    # Summary
    summary = {
        'prescriptions': {
            'total': len(prescr_samples),
            'train': len(prescr_split['train']),
            'test': len(prescr_split['test']),
            'validation': len(prescr_split['validation'])
        },
        'lab_reports': {
            'total': len(lab_samples),
            'train': len(lab_split['train']),
            'test': len(lab_split['test']),
            'validation': len(lab_split['validation'])
        },
        'total_samples': len(prescr_samples) + len(lab_samples)
    }
    
    with open(f"{output_folder}/split_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*70)
    print("✓ SPLIT COMPLETE!")
    print("="*70)
    print(f"\nPRESCRIPTIONS:")
    print(f"  Train: {len(prescr_split['train'])}")
    print(f"  Test:  {len(prescr_split['test'])}")
    print(f"  Val:   {len(prescr_split['validation'])}")
    print(f"\nLAB REPORTS:")
    print(f"  Train: {len(lab_split['train'])}")
    print(f"  Test:  {len(lab_split['test'])}")
    print(f"  Val:   {len(lab_split['validation'])}")
    print(f"\nOutput: {output_folder}\n")
    
    return summary


if __name__ == "__main__":
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        data_folder = "/content/drive/MyDrive/EMR_Data"
        output_folder = "/content/drive/MyDrive/split_data"
    except:
        # Local machine path
        data_folder = "data"
        output_folder = "split_data"
    
    split_dataset(data_folder, output_folder)
