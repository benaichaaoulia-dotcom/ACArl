#!/usr/bin/env python3
"""
List essential files that should be tracked in git for run_pipeline.sh to work.
"""

import os
from pathlib import Path

ESSENTIAL_FILES = {
    "Source Code": [
        "train.py",
        "train_unet.py", 
        "train_unet_acarl.py",
        "lab_gen.py",
        "lab_gen_acarl.py",
        "eval.py",
        "cluster.py",
        "main.py",
    ],
    "Modules": [
        "samus/",
        "samus/modeling/",
        "samus/modeling/acarl.py",
        "brats/",
        "brats/dataset.py",
        "brats/preprocess.py",
        "kits/",
        "kits/dataset.py",
        "kits/preprocess.py",
        "lits/",
        "lits/dataset.py",
        "lits/preprocess.py",
        "unet/",
        "unet/unet_model.py",
        "utils/",
    ],
    "Configuration": [
        "pyproject.toml",
        ".gitignore",
    ],
    "Documentation": [
        "README.md",
    ],
}

def check_files():
    """Check which essential files exist."""
    print("\n" + "="*80)
    print("ESSENTIAL FILES FOR GITHUB PUSH")
    print("="*80 + "\n")
    
    for category, files in ESSENTIAL_FILES.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        for file_pattern in files:
            if "*" in file_pattern or "/" in file_pattern:
                # Directory or pattern
                if "/" in file_pattern:
                    path = Path(file_pattern)
                    exists = path.exists()
                    symbol = "✓" if exists else "✗"
                    print(f"  {symbol} {file_pattern}")
            else:
                # File
                exists = os.path.exists(file_pattern)
                symbol = "✓" if exists else "✗"
                print(f"  {symbol} {file_pattern}")

if __name__ == "__main__":
    check_files()
    
    print("\n" + "="*80)
    print("FILES TO EXCLUDE FROM GIT")
    print("="*80)
    print("""
✗ Datasets (BRATS_Dataset, brats_dataset, etc.)
✗ Checkpoints (checkpoints/ directory)
✗ Logs and results (logs_*, results/)
✗ Preprocessed data (*_preprocessed/)
✗ Generated plots and images (*.png, *.jpg)
✗ Tensorboard events
✗ Shell scripts (run_*.sh, test_*.sh)

Everything is configured in .gitignore
    """)
    
    print("\n" + "="*80)
    print("TO PUSH TO GITHUB")
    print("="*80)
    print("""
1. Initialize git (if not done):
   git init
   
2. Add essential files:
   git add -A
   
3. Verify files to be tracked:
   git status
   
4. Commit:
   git commit -m "Initial commit: WeakMedSAM with ACArL ablation study"
   
5. Add remote:
   git remote add origin <your-repo-url>
   
6. Push:
   git push -u origin main
    """)
