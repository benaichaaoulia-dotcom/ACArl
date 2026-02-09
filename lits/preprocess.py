from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np
import shutil
import nibabel as nib


def preprocess_nifti(ct_dir, mask_dir, output_dir) -> None:
    """
    Preprocess LITS dataset from NIfTI format (3D volumes)
    
    Args:
        ct_dir: Directory containing CT NIfTI files
        mask_dir: Directory containing mask NIfTI files
        output_dir: Output directory for 2D preprocessed images
    """
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii')])
    
    for ct_file in ct_files:
        volume_num = ct_file.replace('.nii', '').replace('volume-', '')
        volume_id = f"volume-{volume_num}"
        
        # Try to find corresponding mask file
        mask_file = os.path.join(mask_dir, f"segmentation-{volume_num}.nii")
        
        if not os.path.exists(mask_file):
            print(f"Warning: No mask found for {ct_file}")
            continue
        
        try:
            # Load NIfTI files
            ct_img = nib.load(os.path.join(ct_dir, ct_file))
            mask_img = nib.load(mask_file)
            
            ct_data = np.asarray(ct_img.dataobj).astype(np.float32)
            mask_data = np.asarray(mask_img.dataobj).astype(np.uint8)
            
            # Create patient output directory
            patient_output_dir = os.path.join(output_dir, volume_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            # Process each slice
            for slice_idx in range(ct_data.shape[2]):
                ct_slice = ct_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]
                
                # Normalize CT to 0-255
                ct_min, ct_max = ct_slice.min(), ct_slice.max()
                if ct_max > ct_min:
                    ct_slice = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
                else:
                    ct_slice = ct_slice.astype(np.uint8)
                
                # Binarize mask: any non-zero is foreground
                mask_slice = (mask_slice > 0).astype(np.uint8) * 255
                
                # Save as 2D images
                Image.fromarray(ct_slice, mode="L").save(
                    os.path.join(patient_output_dir, f"img-{slice_idx:03d}.jpg"), quality=90
                )
                Image.fromarray(mask_slice, mode="L").save(
                    os.path.join(patient_output_dir, f"seg-{slice_idx:03d}.png")
                )
        
        except Exception as e:
            print(f"Error processing {ct_file}: {e}")


def preprocess_jpg(input_dir, output_dir) -> None:
    """
    Preprocess LITS dataset (2D slices from Kaggle JPG format)
    
    Args:
        input_dir: Directory containing JPG images
        output_dir: Output directory
    """
    try:
        # Extract volume ID and slice number from filename
        # Format: volume-X_slice_Y.jpg
        filename = os.path.basename(input_dir)
        parts = filename.replace('.jpg', '').split('_')
        
        if len(parts) >= 3 and parts[0].startswith('volume'):
            volume_id = parts[0]  # volume-X
            patient_name = volume_id
        else:
            patient_name = 'unknown'
        
        # Create patient output directory
        patient_output_dir = os.path.join(output_dir, patient_name)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Load image
        image = Image.open(input_dir).convert("L")
        image = np.array(image, dtype=np.uint8)
        
        # Construct mask path
        mask_path = input_dir.replace('train_images', 'train_masks')
        
        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {input_dir}")
            return
        
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        # Binarize: any non-zero is foreground
        mask[mask != 0] = 255
        
        # Extract slice number from filename
        slice_num = filename.split('_')[-1].replace('.jpg', '')
        
        # Save as individual images
        Image.fromarray(image, mode="L").save(
            os.path.join(patient_output_dir, f"img-{slice_num.zfill(3)}.jpg"), quality=90
        )
        Image.fromarray(mask, mode="L").save(
            os.path.join(patient_output_dir, f"seg-{slice_num.zfill(3)}.png")
        )
        
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")


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
    parser = argparse.ArgumentParser(description="Preprocess LITS dataset")
    parser.add_argument("--input_dir", type=str, default="/home/belal/.cache/kagglehub/datasets/javariatahir/litstrain-val/versions/1/LiTS(train_test)")
    parser.add_argument("--output_dir", type=str, default="./lits_kaggle_preprocessed")
    parser.add_argument("--split_ratio", type=str, default="8:1:1", help="Train:Val:Test split ratio")
    parser.add_argument("--format", type=str, default="nifti", choices=["nifti", "jpg"], help="Input format")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.format == "nifti":
        # Handle NIfTI format
        ct_dir = os.path.join(args.input_dir, "train_CT")
        mask_dir = os.path.join(args.input_dir, "train_mask")
        
        if not os.path.exists(ct_dir) or not os.path.exists(mask_dir):
            print(f"Error: Could not find train_CT or train_mask directories in {args.input_dir}")
            exit(1)
        
        print(f"Processing NIfTI format from {args.input_dir}...")
        preprocess_nifti(ct_dir, mask_dir, args.output_dir)
    
    else:  # jpg format
        # Get all image files
        image_files = sorted([
            os.path.join(args.input_dir, f) 
            for f in os.listdir(args.input_dir) 
            if f.endswith('.jpg')
        ])
        
        print(f"Processing {len(image_files)} LITS images...")
        for img_file in tqdm(image_files):
            preprocess_jpg(img_file, args.output_dir)
    
    print(f"Preprocessing complete! Output: {args.output_dir}")
    generate_splits(args.output_dir, args.split_ratio)
