from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np
import shutil
import nibabel as nib
from multiprocessing import Pool
from functools import partial


def preprocess_single_case(args) -> None:
    """
    Preprocess a single KITS case.
    Handles NIfTI imaging.nii.gz and aggregated_OR_seg.nii.gz segmentation.
    
    Args:
        args: tuple of (case_dir, input_dir, output_dir)
    """
    case_dir, input_dir, output_dir = args
    case_path = os.path.join(input_dir, case_dir)
    
    # Check for NIfTI imaging and aggregated segmentation
    imaging_file = os.path.join(case_path, 'imaging.nii.gz')
    seg_file = os.path.join(case_path, 'aggregated_OR_seg.nii.gz')
    
    if not os.path.exists(imaging_file):
        return f"Skipping {case_dir}: Missing imaging.nii.gz"
    if not os.path.exists(seg_file):
        return f"Skipping {case_dir}: Missing aggregated_OR_seg.nii.gz"
    
    try:
        # Load NIfTI files
        imaging_img = nib.load(imaging_file)
        seg_img = nib.load(seg_file)
        
        imaging_data = np.asarray(imaging_img.dataobj).astype(np.float32)
        seg_data = np.asarray(seg_img.dataobj).astype(np.uint8)
        
        # Ensure same number of slices
        min_slices = min(imaging_data.shape[2], seg_data.shape[2])
        imaging_data = imaging_data[:, :, :min_slices]
        seg_data = seg_data[:, :, :min_slices]
        
        # Create patient output directory
        patient_output_dir = os.path.join(output_dir, case_dir)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Process each slice
        for slice_idx in range(imaging_data.shape[2]):
            imaging_slice = imaging_data[:, :, slice_idx]
            seg_slice = seg_data[:, :, slice_idx]
            
            # Normalize imaging to 0-255
            img_min, img_max = imaging_slice.min(), imaging_slice.max()
            if img_max > img_min:
                imaging_slice = ((imaging_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                imaging_slice = imaging_slice.astype(np.uint8)
            
            # Binarize segmentation: any non-zero is foreground
            seg_slice = (seg_slice > 0).astype(np.uint8) * 255
            
            # Save as 2D images
            Image.fromarray(imaging_slice, mode="L").save(
                os.path.join(patient_output_dir, f"img-{slice_idx:03d}.jpg"), quality=90
            )
            Image.fromarray(seg_slice, mode="L").save(
                os.path.join(patient_output_dir, f"seg-{slice_idx:03d}.png")
            )
        
        return f"Processed {case_dir}"
    
    except Exception as e:
        return f"Error processing {case_dir}: {e}"


def preprocess_kits_cases(input_dir, output_dir, num_workers=None) -> None:
    """
    Preprocess KITS dataset from NIfTI format (3D volumes in case folders) using parallel processing.
    
    Args:
        input_dir: Directory containing case_XXXXX folders with imaging.nii.gz and segmentation.nii.gz
        output_dir: Output directory for 2D preprocessed images
        num_workers: Number of parallel workers (default: auto-detect)
    """
    # Get all case directories
    case_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith('case_')])
    
    # Prepare task arguments
    task_args = [(case_dir, input_dir, output_dir) for case_dir in case_dirs]
    
    # Use multiprocessing
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(preprocess_single_case, task_args), 
                           total=len(task_args),
                           desc="Processing KITS cases"))


def generate_splits(output_path: str, ratio: str = "8:1:1") -> None:
    """Generate patient-based train/val/test splits.

    Args:
        output_path: Path to preprocessed data
        ratio: Split ratio as "train:val:test" string
    """
    splits_dir = os.path.join(output_path, "splits")
    
    # Delete existing splits directory to start fresh
    if os.path.exists(splits_dir):
        shutil.rmtree(splits_dir)
    
    os.makedirs(splits_dir, exist_ok=True)

    # List patient directories AFTER creating splits dir
    # Exclude: hidden files, splits directory itself, and any non-directories
    patients = sorted([
        d for d in os.listdir(output_path) 
        if os.path.isdir(os.path.join(output_path, d)) 
        and not d.startswith('.') 
        and d != 'splits'
    ])
    
    if len(patients) == 0:
        print(f"Warning: No patient directories found in {output_path}")
        return
    
    # Parse ratio
    ratio_parts = [float(x) for x in ratio.split(':')]
    ratio_sum = sum(ratio_parts)
    ratio_parts = [x / ratio_sum for x in ratio_parts]
    
    total = len(patients)
    n_train = int(total * ratio_parts[0])
    n_val = int(total * ratio_parts[1])
    n_test = total - n_train - n_val

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train+n_val]
    test_patients = patients[n_train+n_val:]

    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        for p in train_patients:
            f.write(p + "\n")

    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        for p in val_patients:
            f.write(p + "\n")

    with open(os.path.join(splits_dir, "test.txt"), "w") as f:
        for p in test_patients:
            f.write(p + "\n")

    print(f"Generated splits: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess KITS dataset")
    parser.add_argument("--input_dir", type=str, default="./kits19/data")
    parser.add_argument("--output_dir", type=str, default="./kits_preprocessed")
    parser.add_argument("--split_ratio", type=str, default="8:1:1", help="Train:Val:Test split ratio")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto-detect)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing KITS dataset from {args.input_dir}...")
    print(f"Using {args.workers or 'auto-detected'} workers for parallel processing")
    preprocess_kits_cases(args.input_dir, args.output_dir, args.workers)
    
    print(f"Preprocessing complete! Output: {args.output_dir}")
    generate_splits(args.output_dir, args.split_ratio)
