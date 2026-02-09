#!/usr/bin/env python3
"""
Preprocessing wrapper for flat BRATS dataset structure.
Converts flat structure to HGG-based structure for preprocessing.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

def main():
    parser = argparse.ArgumentParser(description="Preprocess flat BRATS dataset")
    parser.add_argument("--input-path", type=str, help="Input dataset path (flat structure)")
    parser.add_argument("--output-path", type=str, help="Output preprocessed data path")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    # Check if input is flat structure
    brats_dirs = sorted([d for d in input_path.iterdir() if d.is_dir() and d.name.startswith('BraTS')])
    
    if not brats_dirs:
        print(f"Error: No BraTS directories found in {input_path}", file=sys.stderr)
        return 1
    
    # Check if already structured (has HGG/LGG)
    has_hgg = (input_path / 'HGG').exists()
    has_lgg = (input_path / 'LGG').exists()
    
    if has_hgg or has_lgg:
        print(f"Dataset already structured with HGG/LGG directories")
        input_for_preprocess = input_path
    else:
        print(f"Flat dataset detected with {len(brats_dirs)} patients")
        print(f"Creating temporary structured format...")
        
        # Create temporary structured directory
        temp_dir = tempfile.mkdtemp(prefix="brats_structured_")
        temp_hgg = Path(temp_dir) / 'HGG'
        temp_lgg = Path(temp_dir) / 'LGG'
        temp_hgg.mkdir(parents=True, exist_ok=True)
        temp_lgg.mkdir(parents=True, exist_ok=True)
        
        # Link all patients under HGG (LGG will stay empty)
        for patient_dir in brats_dirs:
            target = temp_hgg / patient_dir.name
            if not target.exists():
                target.symlink_to(patient_dir)
        
        input_for_preprocess = Path(temp_dir)
        print(f"Using temporary structured path: {temp_dir}")
        print(f"  - HGG: {len(brats_dirs)} patients")
        print(f"  - LGG: 0 patients")
    
    # Now run the actual preprocessing
    print(f"\nRunning BRATS preprocessing...")
    print(f"  Input: {input_for_preprocess}")
    print(f"  Output: {output_path}")
    print(f"  Workers: {args.workers}")
    
    # Call the original BRATS preprocess script
    result = subprocess.run([
        sys.executable, "brats/preprocess.py",
        "--input-path", str(input_for_preprocess),
        "--output-path", str(output_path),
        "--workers", str(args.workers)
    ])
    
    # Cleanup temp directory if created
    if not has_hgg and not has_lgg:
        print(f"\nCleaning up temporary directory...")
        shutil.rmtree(input_for_preprocess, ignore_errors=True)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
