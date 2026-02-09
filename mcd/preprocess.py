import SimpleITK as sitk
from PIL import Image
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


def preprocess(args) -> None:
    """
    Preprocess a single MCD Heart volume (nii.gz) into 2D slices
    Saves slices into patient-based directory structure
    
    Args:
        args: (input_file, output_dir) tuple where input_file is the image nii.gz path
    """
    input_file, output_dir = args
    
    try:
        # Extract patient name from filename (e.g., 'la_003_0000.nii.gz' -> 'la_003')
        filename = os.path.basename(input_file)
        patient_name = filename.split('_0000')[0] if '_0000' in filename else filename.split('.nii.gz')[0]
        
        # Create patient output directory
        patient_output_dir = os.path.join(output_dir, patient_name)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Load image volume
        image = sitk.GetArrayFromImage(sitk.ReadImage(input_file)).astype(np.float32)
        
        # Normalize to 0-255
        if image.size > 0:
            image_min = image.min()
            image_max = image.max()
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        
        # Load label volume
        # Task02_Heart uses naming: la_014.nii.gz (image) and la_014.nii.gz (label in labelsTr folder)
        # So we need to construct the path based on the directory structure
        label_file = input_file.replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz')
        
        # If file not found, try alternative naming with _seg suffix
        if not os.path.exists(label_file):
            base = input_file.replace('imagesTr', 'labelsTr')
            label_file = base.replace('_0000.nii.gz', '_seg.nii.gz') if '_0000' in base else base.replace('.nii.gz', '_seg.nii.gz')
        
        if not os.path.exists(label_file):
            print(f"Warning: No segmentation file found for {patient_name}")
            print(f"  Tried: {label_file}")
            return
        
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_file)).astype(np.uint8)
        # Binarize: any non-zero is foreground
        label[label != 0] = 255
        
        # Save slices
        for i, (img, lab) in enumerate(zip(image, label)):
            Image.fromarray(img, mode="L").save(
                os.path.join(patient_output_dir, f"img-{str(i).zfill(3)}.jpg"), quality=90
            )
            Image.fromarray(lab, mode="L").save(
                os.path.join(patient_output_dir, f"seg-{str(i).zfill(3)}.png")
            )
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def generate_splits(output_path: str) -> None:
    """Generate patient-based train/val/test splits in output_path/splits.

    Creates train.txt (80%), val.txt (20%), and empty test.txt as requested.
    """
    splits_dir = os.path.join(output_path, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # list patient directories (ignore hidden files)
    patients = sorted([d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and not d.startswith('.')])
    patient_array = patients
    total = len(patient_array)
    n_train = int(total * 80 / 100)

    train_patients = patient_array[:n_train]
    val_patients = patient_array[n_train:]

    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        for p in train_patients:
            f.write(p + "\n")

    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        for p in val_patients:
            f.write(p + "\n")

    # empty test.txt
    open(os.path.join(splits_dir, "test.txt"), "w").close()

    print(f"Generated splits in {splits_dir}: {len(train_patients)} train, {len(val_patients)} val, 0 test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCD Heart preprocessing")
    parser.add_argument("--input-path", type=str, help="input path to imagesTr/imagesTs", required=False)
    parser.add_argument("--output-path", type=str, help="output path", required=True)
    parser.add_argument("--generate-splits", action="store_true", help="Only generate train/val/test splits in the output path")
    parser.add_argument("--workers", type=int, help="preprocess workers", default=4)
    args = parser.parse_args()

    # Ensure output path exists (do not fail if it already exists)
    os.makedirs(args.output_path, exist_ok=True)

    # If not running split-only mode, require input path
    if not args.generate_splits:
        if not args.input_path or not os.path.exists(args.input_path):
            print(f"Input path does not exist or not provided: {args.input_path}")
            raise SystemExit(1)

    # Collect all NIfTI image files (not segmentation files)
    input_files = []
    
    if args.input_path and os.path.exists(args.input_path):
        for filename in sorted(os.listdir(args.input_path)):
            # Only process image files, not segmentation files
            if filename.endswith('.nii.gz') and '_seg' not in filename:
                input_files.append(os.path.join(args.input_path, filename))

    print(f"Found {len(input_files)} samples to preprocess")

    with mp.Pool(args.workers) as pool:
        list(
            tqdm(
                pool.imap(preprocess, [(f, args.output_path) for f in input_files]),
                total=len(input_files),
                ncols=100,
            )
        )

    print(f"Preprocessing complete. Output saved to {args.output_path}")
    # Generate splits after preprocessing
    generate_splits(args.output_path)

