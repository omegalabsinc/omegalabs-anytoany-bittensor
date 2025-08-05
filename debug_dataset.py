#!/usr/bin/env python3
"""
Debug VoiceBench dataset structure
"""

import sys
sys.path.insert(0, '/home/salman/anmol/VoiceBench')

from datasets import load_dataset, Audio

# Test loading a dataset to see its structure
dataset = load_dataset('hlt-lab/voicebench', 'alpacaeval', split='test')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

print("Dataset info:")
print(f"Length: {len(dataset)}")
print(f"Features: {dataset.features}")
print(f"Column names: {dataset.column_names}")

# Check first item
if len(dataset) > 0:
    item = dataset[0]
    print("\nFirst item structure:")
    for key, value in item.items():
        if key == 'audio':
            print(f"  {key}: {type(value)} - keys: {value.keys() if hasattr(value, 'keys') else 'not a dict'}")
            if hasattr(value, 'keys'):
                for k, v in value.items():
                    print(f"    {k}: {type(v)} - shape: {v.shape if hasattr(v, 'shape') else len(v) if hasattr(v, '__len__') else 'N/A'}")
        else:
            print(f"  {key}: {type(value)} - {str(value)[:100]}")